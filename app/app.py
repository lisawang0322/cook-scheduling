"""7-Eleven Cook Scheduling — Associate Tablet Interface

A self-contained screen simulating the hand-held tablet experience used by store associates.
Includes color-coded queue, timing metrics, explanation, override controls, and cook confirmation.
"""

import os
import sys
import time
from datetime import datetime
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import (
    load_model, load_labeled_data, get_v22_ranking, get_v1_ranking,
    compute_queue_timing, generate_explanation, log_associate_action,
    ITEM_DISPLAY_NAMES, ITEM_EMOJIS, STORE_TYPE_LABELS, HOUR_LABELS, OVEN_ITEMS
)

st.set_page_config(
    page_title="7-Eleven Hot Food Assistant",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling for Tablet UI ---
st.markdown("""
<style>
    .tablet-header {
        background-color: #008060;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .tablet-title {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }
    .tablet-subtitle {
        font-size: 16px;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    .queue-card {
        border-left: 10px solid #ccc;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .card-now { border-left-color: #e63946; background-color: #fff1f2; }
    .card-next { border-left-color: #ffb703; background-color: #fffbeb; }
    .card-then { border-left-color: #2a9d8f; background-color: #f0fdfa; }
    .card-skip { border-left-color: #6c757d; background-color: #f3f4f6; opacity: 0.7; }
    
    .card-title {
        font-size: 22px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 8px;
    }
    .card-meta {
        font-size: 14px;
        color: #4b5563;
        line-height: 1.6;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: bold;
        border-radius: 4px;
        color: white;
        margin-bottom: 10px;
    }
    .badge-now { background-color: #e63946; }
    .badge-next { background-color: #ffb703; color: #1f2937; }
    .badge-then { background-color: #2a9d8f; }
    .badge-skip { background-color: #6c757d; }
    .badge-fallback { background-color: #d97706; margin-left: 10px; }
    
    .grill-box {
        background-color: #eff6ff;
        border: 2px dashed #3b82f6;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
    .explanation-box {
        background-color: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Model ---
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_data():
    return load_labeled_data()

try:
    model, historical = get_model()
    labeled_data = get_data()
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.stop()

# --- Initialize Session State ---
if "history_idx" not in st.session_state:
    st.session_state["history_idx"] = 0

# Load a scenario
if "current_scenario" not in st.session_state or st.session_state.get("needs_reload", False):
    # Retrieve scenario
    scenarios = labeled_data
    idx = st.session_state["history_idx"] % len(scenarios)
    scenario = scenarios[idx]
    
    st.session_state["current_scenario"] = scenario
    st.session_state["needs_reload"] = False
    st.session_state["confirmed"] = False
    st.session_state["skipped_items"] = []
    
    # Compute initial rankings
    features = scenario["features"]
    v22_ranking = get_v22_ranking(features, model, historical)
    st.session_state["original_order"] = list(v22_ranking)
    st.session_state["working_order"] = list(v22_ranking)
    st.session_state["fallback_active"] = False

scenario = st.session_state["current_scenario"]
features = scenario["features"]
decision_hour = features["decision_hour"]
store_type = features["store_type"]
day_of_week = features["day_of_week"]

# --- Sidebar Controls (For Demo Purposes Only) ---
st.sidebar.title("🎛️ Demo Controls")
st.sidebar.markdown("Use these to simulate changes or advance the demo.")

# Scenario selection
st.sidebar.markdown("### Decision Points")
if st.sidebar.button("➡️ Next Decision Point"):
    st.session_state["history_idx"] += 1
    st.session_state["needs_reload"] = True
    st.rerun()

if st.sidebar.button("🔄 Reset Current"):
    st.session_state["needs_reload"] = True
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Failure-Mode Simulation")
simulate_low_confidence = st.sidebar.checkbox(
    "⚠️ Simulate Low ML Confidence (<70%)",
    value=st.session_state.get("simulate_low_confidence", False),
    help="Triggers fallback to deterministic v1 rules."
)
st.session_state["simulate_low_confidence"] = simulate_low_confidence

# Force recalculation if low confidence mode toggled
if simulate_low_confidence and not st.session_state["fallback_active"]:
    st.session_state["working_order"] = get_v1_ranking(features)
    st.session_state["fallback_active"] = True
elif not simulate_low_confidence and st.session_state["fallback_active"]:
    st.session_state["working_order"] = list(st.session_state["original_order"])
    st.session_state["fallback_active"] = False

# Display Active Log
st.sidebar.markdown("---")
st.sidebar.markdown("### System State")
st.sidebar.info(
    f"Active Model: {'Rule-Based Fallback (v1)' if st.session_state['fallback_active'] else 'GradientBoosting (v2.2)'}\n\n"
    f"Overrides Logged: {len([x for x in os.listdir(os.path.join(PROJECT_ROOT, 'output')) if 'overrides' in x]) > 0}"
)

# --- Tablet Header ---
st.markdown(
    f"""
    <div class="tablet-header">
        <div class="tablet-title">🏪 7-Eleven Hot Food Assistant</div>
        <div class="tablet-subtitle">
            Store Type: {STORE_TYPE_LABELS.get(store_type, store_type).split(' (')[0]} &nbsp;|&nbsp; 
            Shift: {day_of_week} at {HOUR_LABELS.get(decision_hour, f'{decision_hour}:00')}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state["confirmed"]:
    st.balloons()
    st.success("🎉 **Cook Confirmed!** Recommendation has been sent to the oven. Override action logged successfully.")
    st.markdown("### Next Steps:")
    st.markdown("1. Load the recommended quantities of food into the oven trays.")
    st.markdown("2. Press the start button on the oven for the corresponding shelf program.")
    st.markdown("3. Tap **Next Decision Point** in the sidebar to simulate the next cooking cycle.")
    
    if st.button("Start Next Cycle"):
        st.session_state["history_idx"] += 1
        st.session_state["needs_reload"] = True
        st.rerun()
    st.stop()

# --- Main Tablet Screen ---
col_queue, col_controls = st.columns([3, 1])

with col_queue:
    st.markdown("### 📋 Oven Cook Queue")
    st.caption("Perform your tasks in this order. Use the controls on the right if you need to adjust.")
    
    working_order = st.session_state["working_order"]
    skipped_items = st.session_state["skipped_items"]
    
    if not working_order and not skipped_items:
        st.warning("No items in queue.")
    
    # 1. Render active oven queue items
    for i, item in enumerate(working_order):
        # Determine styling
        if i == 0:
            card_class = "card-now"
            badge_class = "badge-now"
            badge_text = "🔴 COOK NOW"
        elif i == 1:
            card_class = "card-next"
            badge_class = "badge-next"
            badge_text = "🟡 NEXT"
        else:
            card_class = "card-then"
            badge_class = "badge-then"
            badge_text = "🟢 THEN"
            
        # Get timing calculations
        timing = compute_queue_timing(item, decision_hour)
        emoji = ITEM_EMOJIS.get(item, "")
        display_name = ITEM_DISPLAY_NAMES.get(item, item)
        demand = features.get(f"{item}_forecast_demand", 0)
        lcu = features.get(f"{item}_lcu", 1)
        batches = max(1, demand // lcu)
        
        fallback_badge = ""
        if st.session_state["fallback_active"] and i == 0:
            fallback_badge = '<span class="badge badge-fallback">⚠️ FALLBACK RULES ACTIVE</span>'
            
        st.markdown(
            f"""
            <div class="queue-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="badge {badge_class}">{badge_text}</span>
                        {fallback_badge}
                        <div class="card-title">{emoji} {display_name}</div>
                    </div>
                    <div style="text-align: right; font-size: 18px; font-weight: bold; color: #008060;">
                        Cook Qty: {demand} units<br>
                        <span style="font-size: 12px; color: #6b7280; font-weight: normal;">
                            ({batches} batch{'es' if batches != 1 else ''} of {lcu})
                        </span>
                    </div>
                </div>
                <div class="card-meta">
                    ⏱️ <b>Cook Time:</b> {timing['cook_time_mins']} mins &nbsp;|&nbsp; 
                    🕒 <b>Oven Program Ready:</b> {timing['ready_time_str']} &nbsp;|&nbsp; 
                    ⌛ <b>Hold Time:</b> {timing['hold_time_hours']} hrs &nbsp;|&nbsp; 
                    🚨 <b>Discard Expiry:</b> {timing['expiry_time_str']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. Render skipped items (if any)
    if skipped_items:
        st.markdown("#### ⏳ Deferred / Skipped Items")
        for item in skipped_items:
            emoji = ITEM_EMOJIS.get(item, "")
            display_name = ITEM_DISPLAY_NAMES.get(item, item)
            st.markdown(
                f"""
                <div class="queue-card card-skip">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="badge badge-skip">⚪ SKIPPED</span>
                            <div class="card-title" style="font-size: 18px; margin: 0;">{emoji} {display_name}</div>
                        </div>
                        <div class="card-meta" style="font-size: 12px;">
                            Deferred for this cooking cycle.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 3. Grill Items Note
    st.markdown(
        f"""
        <div class="grill-box">
            <h4 style="margin: 0 0 8px 0; color: #1e3a8a;">� Grill Items (Scheduled)</h4>
            <div style="font-size: 14px; color: #1e40af;">
                <b>Hot dog, sausage, taquito, buffalo roller, and corn dog</b> use the roller grill
                and are included in the scheduling queue.
                <br>⏱️ <b>Grill Cook Time:</b> 20 mins | ⌛ <b>Hold Time:</b> 4 hrs
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_controls:
    st.markdown("### 🎛️ Queue Actions")
    st.caption("Adjust items or confirm cook")
    
    # 1. Override and Reordering section
    st.markdown("#### Move / Adjust")
    
    for i, item in enumerate(working_order):
        display_name = ITEM_DISPLAY_NAMES.get(item, item)
        emoji = ITEM_EMOJIS.get(item, "")
        
        st.markdown(f"**{emoji} {display_name}**")
        btn_cols = st.columns(3)
        
        # Up button
        with btn_cols[0]:
            if i > 0:
                if st.button("▲ Up", key=f"btn_up_{item}_{i}"):
                    # Swap
                    st.session_state["working_order"][i], st.session_state["working_order"][i-1] = \
                        st.session_state["working_order"][i-1], st.session_state["working_order"][i]
                    st.rerun()
            else:
                st.button("▲ Up", key=f"btn_up_disabled_{item}_{i}", disabled=True)
                
        # Down button
        with btn_cols[1]:
            if i < len(working_order) - 1:
                if st.button("▼ Dn", key=f"btn_dn_{item}_{i}"):
                    # Swap
                    st.session_state["working_order"][i], st.session_state["working_order"][i+1] = \
                        st.session_state["working_order"][i+1], st.session_state["working_order"][i]
                    st.rerun()
            else:
                st.button("▼ Dn", key=f"btn_dn_disabled_{item}_{i}", disabled=True)
                
        # Skip button
        with btn_cols[2]:
            if st.button("✕ Skp", key=f"btn_skip_{item}_{i}"):
                st.session_state["working_order"].remove(item)
                st.session_state["skipped_items"].append(item)
                st.rerun()
                
        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
        
    # Unskip controls
    if skipped_items:
        st.markdown("#### Restore Items")
        for item in skipped_items:
            display_name = ITEM_DISPLAY_NAMES.get(item, item)
            if st.button(f"Restore {display_name}", key=f"restore_{item}"):
                st.session_state["skipped_items"].remove(item)
                st.session_state["working_order"].append(item)
                st.rerun()

    st.markdown("---")
    
    # 2. Main Confirm Button
    st.markdown("#### Finalize Task")
    
    is_altered = st.session_state["working_order"] != st.session_state["original_order"] or len(skipped_items) > 0
    
    if is_altered:
        st.warning("⚠️ You have modified the recommended order.")
    
    confirm_btn = st.button("🟢 CONFIRM COOK NOW", use_container_width=True, type="primary")
    
    if confirm_btn:
        # Build action log entry
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "scenario_index": st.session_state["history_idx"],
            "store_type": store_type,
            "decision_hour": decision_hour,
            "day_of_week": day_of_week,
            "is_override": is_altered,
            "fallback_active": st.session_state["fallback_active"],
            "original_order": st.session_state["original_order"],
            "final_order": st.session_state["working_order"],
            "skipped_items": skipped_items,
        }
        
        # Log to file
        log_associate_action(log_entry)
        
        # Mark as confirmed and rerun
        st.session_state["confirmed"] = True
        st.rerun()

# --- Explanation Section ---
st.markdown("---")
st.markdown("### 💡 Why is this order recommended?")

explanations = generate_explanation(working_order, features)
st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
for exp in explanations:
    st.markdown(f"- {exp}")
st.markdown("</div>", unsafe_allow_html=True)

# Add clear validation metrics from cost model / ROI
st.caption(
    "Ground Truth Guidance: Following the recommended sequence ensures hot-bar availability while minimizing "
    "waste. Historically, this model delivers +19.3 pp accuracy lift over random guessing, saving an estimated "
    "$180/store/month in write-offs (based on CostModel_lisaw2.xlsx parameters)."
)
