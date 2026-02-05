import streamlit as st
import requests
import json
import pandas as pd
import os
import time
import google.generativeai as genai
from datetime import datetime, timezone

# ----------------------------
# CONFIGURATION & SECRETS
# ----------------------------

def get_secret(key: str):
    # Prefer Streamlit Community Cloud secrets, fallback to env vars for local runs.
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

UPSTASH_URL = get_secret("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = get_secret("UPSTASH_REDIS_REST_TOKEN")
GEMINI_KEY = get_secret("GEMINI_API_KEY")

# Model configuration
# You saw "Gemini 3 Flash" in AI Studio model list, so use gemini-3-flash here.
# If your SDK expects a different exact string, change ONLY this line.
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# Redis Stream key
STREAM_KEY = "telemetry:stream"


# ----------------------------
# PROMPT (Single-call EE + ACE)
# ----------------------------

SINGLE_CALL_PROMPT = """
You are Project Forge, an industrial self-healing digital twin agent.

You will receive recent sensor readings for equipment 'arm_12' as readings_json.

Baselines:
- temperature_c: 45-55
- vibration_hz: 20-30
- voltage_v: 220-240

Failure Modes:
A) overheat_friction: temperature_c > 60 AND vibration_hz > 40 AND trend == increasing
B) power_brownout: voltage_v < 210 AND temperature trend == decreasing
Otherwise: unknown or null

Playbook:
- rule_overheat_001: if failure_mode == overheat_friction AND confidence >= 80 -> reduce_arm_speed to 70%
- rule_brownout_001: if failure_mode == power_brownout AND confidence >= 90 -> trigger_backup_power
- else -> continue_monitoring

REQUIREMENTS:
1) Compute simple trends from the readings (increasing/decreasing) using last 5 points if available.
2) Produce strict JSON ONLY. No markdown. No extra text.
3) Output must match this schema exactly:

{
  "mode": "gemini",
  "ee_result": {
    "equipment_id": "arm_12",
    "anomaly_detected": true/false,
    "failure_mode": "overheat_friction" | "power_brownout" | "unknown" | null,
    "confidence": 0-100,
    "severity": "low" | "medium" | "high" | "critical",
    "deviating_sensors": ["temperature_c","vibration_hz","voltage_v"],
    "recommended_action": "reduce_arm_speed" | "trigger_cooling" | "trigger_backup_power" | "alert_operator" | "continue_monitoring",
    "reasoning": "one short sentence"
  },
  "ace_result": {
    "matched_rules": ["rule_overheat_001"],
    "confidence": 0-100,
    "decision": "execute_action" | "continue_monitoring",
    "action_payload": { "equipment_id": "arm_12", "command": "string", "parameters": {} } | null,
    "maintenance_event_log": {
      "event_id": "string",
      "equipment_id": "arm_12",
      "timestamp": "string",
      "failure_mode": "string",
      "action_taken": "string",
      "confidence": 0-100,
      "rule_applied": "string"
    }
  }
}

Now here is readings_json:
"""


# ----------------------------
# UTILITIES: Redis Stream
# ----------------------------

def parse_redis_stream(raw_data: dict):
    """Parse Upstash XREVRANGE response into list of dicts (sorted asc by ts)."""
    parsed = []
    if not isinstance(raw_data, dict) or "result" not in raw_data:
        return parsed

    for entry in raw_data["result"]:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        entry_id = entry[0]
        fields = entry[1] if isinstance(entry[1], list) else []

        d = {"id": str(entry_id)}
        for i in range(0, len(fields), 2):
            if i + 1 >= len(fields):
                break
            key = fields[i]
            val = fields[i + 1]
            try:
                if key != "ts":
                    d[key] = float(val)
                else:
                    d[key] = val
            except Exception:
                d[key] = val

        # Derive ts if missing
        if "ts" not in d:
            try:
                ms = int(str(entry_id).split("-")[0])
                d["ts"] = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
            except Exception:
                d["ts"] = str(entry_id)

        parsed.append(d)

    parsed.sort(key=lambda x: x.get("ts", ""))
    return parsed


def fetch_stream_last_n(n: int):
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        raise RuntimeError("Missing UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN.")

    headers = {"Authorization": f"Bearer {UPSTASH_TOKEN}"}
    body = ["XREVRANGE", STREAM_KEY, "+", "-", "COUNT", int(n)]
    resp = requests.post(UPSTASH_URL, headers=headers, json=body, timeout=20)
    resp.raise_for_status()
    return parse_redis_stream(resp.json())


# ----------------------------
# UTILITIES: Gemini + Fallback
# ----------------------------

def _now_iso():
    return datetime.now(tz=timezone.utc).isoformat()

def local_fallback(readings):
    """
    Deterministic fallback that mimics the EE + ACE outputs when Gemini is rate-limited.
    """
    if not readings:
        return {"error": "No readings for fallback."}

    latest = readings[-1]
    temp = float(latest.get("temperature_c", 0) or 0)
    vib = float(latest.get("vibration_hz", 0) or 0)
    volt = float(latest.get("voltage_v", 0) or 0)

    last5 = readings[-5:] if len(readings) >= 5 else readings

    def trend(key):
        vals = []
        for r in last5:
            if key in r:
                try:
                    vals.append(float(r[key]))
                except Exception:
                    pass
        if len(vals) < 2:
            return "unknown"
        if vals[-1] > vals[0]:
            return "increasing"
        if vals[-1] < vals[0]:
            return "decreasing"
        return "flat"

    temp_trend = trend("temperature_c")
    vib_trend = trend("vibration_hz")

    deviating = []
    if temp < 45 or temp > 55:
        deviating.append("temperature_c")
    if vib < 20 or vib > 30:
        deviating.append("vibration_hz")
    if volt < 220 or volt > 240:
        deviating.append("voltage_v")

    anomaly_detected = len(deviating) > 0

    failure_mode = None
    confidence = 60
    severity = "low"
    recommended_action = "continue_monitoring"

    if temp > 60 and vib > 40 and (temp_trend == "increasing" or vib_trend == "increasing"):
        failure_mode = "overheat_friction"
        confidence = 85
        severity = "high" if temp > 65 or vib > 50 else "medium"
        recommended_action = "reduce_arm_speed"
    elif volt < 210 and temp_trend == "decreasing":
        failure_mode = "power_brownout"
        confidence = 90
        severity = "high"
        recommended_action = "trigger_backup_power"
    else:
        failure_mode = "unknown" if anomaly_detected else None
        confidence = 55 if anomaly_detected else 40
        severity = "medium" if anomaly_detected and (temp > 58 or vib > 35 or volt < 215) else "low"
        recommended_action = "continue_monitoring"

    ee_result = {
        "equipment_id": "arm_12",
        "anomaly_detected": bool(anomaly_detected),
        "failure_mode": failure_mode,
        "confidence": confidence,
        "severity": severity,
        "deviating_sensors": deviating,
        "recommended_action": recommended_action,
        "reasoning": "Fallback rules applied due to Gemini rate limit/quota.",
    }

    matched_rules = []
    ace_conf = confidence
    decision = "continue_monitoring"
    action_payload = None
    rule_applied = ""
    action_taken = "continue_monitoring"

    if failure_mode == "overheat_friction" and confidence >= 80:
        matched_rules = ["rule_overheat_001"]
        decision = "execute_action"
        rule_applied = "rule_overheat_001"
        action_taken = "reduce_arm_speed"
        action_payload = {"equipment_id": "arm_12", "command": "set_speed", "parameters": {"target_speed_percent": 70}}
    elif failure_mode == "power_brownout" and confidence >= 90:
        matched_rules = ["rule_brownout_001"]
        decision = "execute_action"
        rule_applied = "rule_brownout_001"
        action_taken = "trigger_backup_power"
        action_payload = {"equipment_id": "arm_12", "command": "trigger_backup_power", "parameters": {}}

    ace_result = {
        "matched_rules": matched_rules,
        "confidence": ace_conf,
        "decision": decision,
        "action_payload": action_payload,
        "maintenance_event_log": {
            "event_id": f"evt_{int(time.time())}",
            "equipment_id": "arm_12",
            "timestamp": _now_iso(),
            "failure_mode": str(failure_mode),
            "action_taken": action_taken,
            "confidence": ace_conf,
            "rule_applied": rule_applied,
        },
    }

    return {"mode": "fallback", "ee_result": ee_result, "ace_result": ace_result}


def get_gemini_or_fallback(readings):
    """
    Try Gemini once; if quota/429 occurs, return deterministic fallback.
    Adds caching based on latest stream entry id.
    """
    if not readings:
        return {"error": "No readings to analyze."}

    latest_id = readings[-1].get("id", "")
    cache_key = f"result::{latest_id}"

    if "result_cache" not in st.session_state:
        st.session_state.result_cache = {}

    if cache_key in st.session_state.result_cache:
        cached = dict(st.session_state.result_cache[cache_key])
        cached["_cached"] = True
        return cached

    # Minimize payload
    readings_min = [
        {
            "ts": r.get("ts"),
            "temperature_c": r.get("temperature_c"),
            "vibration_hz": r.get("vibration_hz"),
            "voltage_v": r.get("voltage_v"),
            "id": r.get("id"),
        }
        for r in readings
    ]

    prompt = SINGLE_CALL_PROMPT + "\n" + json.dumps(readings_min, ensure_ascii=False) + "\n\nReturn JSON only."

    if not GEMINI_KEY:
        res = {"error": "GEMINI_API_KEY missing"}
        st.session_state.result_cache[cache_key] = res
        return res

    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        text = getattr(response, "text", "") or ""
        result = json.loads(text)

        if isinstance(result, dict) and "mode" not in result:
            result["mode"] = "gemini"

        st.session_state.result_cache[cache_key] = result
        return result

    except Exception as e:
        msg = str(e)

        # Detect 429/quota/rate limit cases
        if ("429" in msg) or ("quota" in msg.lower()) or ("rate" in msg.lower()):
            fb = local_fallback(readings)
            fb["_note"] = "Gemini quota/rate limited; fallback used."
            fb["_gemini_error"] = msg  # <<<<<< show raw error for debugging
            st.session_state.result_cache[cache_key] = fb
            return fb

        err = {"error": msg}
        st.session_state.result_cache[cache_key] = err
        return err


def simulate_data(n):
    data = []
    now = time.time()
    for i in range(n):
        data.append(
            {
                "id": f"{int((now - i * 5) * 1000)}-0",
                "ts": datetime.fromtimestamp(now - i * 5, tz=timezone.utc).isoformat(),
                "temperature_c": 50 + (15 if i < 3 else 0),
                "vibration_hz": 25 + (20 if i < 3 else 0),
                "voltage_v": 230 + (i % 5),
            }
        )
    data.sort(key=lambda x: x["ts"])
    return data


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Project Forge", layout="wide")
st.title("Project Forge — Self-Healing Industrial Brain")
st.markdown("Redis Stream (Upstash) → **Gemini EE+ACE** (single call) → Action payload")

if "readings" not in st.session_state:
    st.session_state.readings = []
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.header("Controls")
    n_entries = st.number_input("Entries to fetch", 5, 200, 20)
    auto_refresh = st.checkbox("Auto-refresh (5s)")

    if st.button("Fetch Stream Data", use_container_width=True):
        try:
            st.session_state.readings = fetch_stream_last_n(int(n_entries))
            if not st.session_state.readings:
                st.warning("Stream is empty or could not be parsed.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    if st.button("Generate Simulated Data", use_container_width=True):
        st.session_state.readings = simulate_data(int(n_entries))
        st.success("Simulated data generated.")

    st.divider()

    if st.button("Run EE+ACE (Single Call)", type="primary", use_container_width=True):
        if not st.session_state.readings:
            st.warning("No data to analyze. Fetch stream data first.")
        else:
            with st.spinner("Running EE+ACE..."):
                st.session_state.result = get_gemini_or_fallback(st.session_state.readings)

if st.button("List Gemini Models (debug)", use_container_width=True):
    if not GEMINI_KEY:
        st.error("GEMINI_API_KEY missing")
    else:
        try:
            genai.configure(api_key=GEMINI_KEY)
            models = list(genai.list_models())
            rows = []
            for m in models:
                # m.name looks like "models/..."
                # m.supported_generation_methods might include "generateContent"
                rows.append({
                    "name": getattr(m, "name", ""),
                    "display_name": getattr(m, "display_name", ""),
                    "methods": ", ".join(getattr(m, "supported_generation_methods", []) or []),
                })
            st.session_state["_models_debug"] = rows
            st.success(f"Found {len(rows)} models.")
        except Exception as e:
            st.error(str(e))


# Main View
col1, col2, col3 = st.columns([1.2, 1, 1])
if st.session_state.get("_models_debug"):
    with st.expander("Gemini models (from ListModels)"):
        st.dataframe(pd.DataFrame(st.session_state["_models_debug"]), use_container_width=True)

with col1:
    st.subheader("Raw Stream Telemetry")
    if st.session_state.readings:
        df = pd.DataFrame(st.session_state.readings)
        st.dataframe(df, use_container_width=True)
        with st.expander("View Raw JSON"):
            st.json(st.session_state.readings)
    else:
        st.info("No data fetched yet.")

with col2:
    st.subheader("EE Result")
    res = st.session_state.result
    if res:
        if "error" in res:
            st.error(res["error"])
        else:
            mode = res.get("mode", "gemini")
            st.caption(f"mode: {mode}" + (" (cached)" if res.get("_cached") else ""))
            ee = res.get("ee_result", {})
            if ee:
                sev = str(ee.get("severity", "low")).lower()
                color = {"critical": "red", "high": "orange", "medium": "yellow", "low": "blue"}.get(sev, "gray")
                st.markdown(f"**Severity:** :{color}[{sev.upper()}]")
                st.json(ee)
            else:
                st.warning("No ee_result in response.")

            if res.get("_note"):
                st.info(res["_note"])

            # Show raw Gemini error (debug)
            if res.get("_gemini_error"):
                with st.expander("Show Gemini raw error"):
                    st.code(res["_gemini_error"])
    else:
        st.info("Run EE+ACE to see results.")

with col3:
    st.subheader("ACE Decision")
    res = st.session_state.result
    if res and "error" not in res:
        ace = res.get("ace_result", {})
        if ace:
            decision = str(ace.get("decision", "continue_monitoring")).lower()
            st.markdown(f"**Decision:** :{'green' if decision == 'execute_action' else 'gray'}[{decision.upper()}]")
            st.json(ace)
        else:
            st.warning("No ace_result in response.")
    else:
        st.info("Run EE+ACE to see results.")

if auto_refresh:
    time.sleep(5)
    st.rerun()
