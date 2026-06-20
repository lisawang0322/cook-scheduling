"""Page 3: What-If Simulator

Adjust inputs and see how the ML recommendation changes in real-time.
"""

import streamlit as st
import sys
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils import (
    load_model, get_v1_ranking, get_associate_pick, get_v22_ranking,
    generate_explanation, generate_forecast_demand,
    ITEM_DISPLAY_NAMES, ITEM_EMOJIS,
    STORE_TYPE_LABELS, HOUR_LABELS, OVEN_ITEMS,
)

st.set_page_config(page_title="What-If Simulator", page_icon="🎛️", layout="wide")

st.title("🎛️ What-If Simulator")
st.markdown("Adjust store conditions and see how the recommendation changes in real-time.")

# --- Load model ---
@st.cache_resource
def get_model():
    return load_model()

model, historical = get_model()

# --- Input Controls ---
st.sidebar.markdown("### Store Conditions")

store_type = st.sidebar.selectbox(
    "Store Type",
    ["urban", "suburban", "highway"],
    format_func=lambda x: STORE_TYPE_LABELS[x],
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)

hour = st.sidebar.slider("Hour", 6, 23, 12,
                          format="%d:00")

is_weekend = day_of_week in ("Saturday", "Sunday")

st.sidebar.markdown("---")
st.sidebar.markdown("### Items in Oven Queue")
st.sidebar.caption("Toggle which items are competing for the oven")

# Item toggles — demand is auto-generated from context above
item_config = {}
for item in OVEN_ITEMS:
    emoji = ITEM_EMOJIS.get(item, "")
    display = ITEM_DISPLAY_NAMES[item]
    enabled = st.sidebar.checkbox(f"{emoji} {display}", value=(item != "wings_4h"))
    if enabled:
        demand = generate_forecast_demand(item, hour, store_type, is_weekend)
        item_config[item] = demand

# --- Build features ---
if len(item_config) < 2:
    st.warning("⚠️ Select at least 2 items in the sidebar to see a ranking comparison.")
    st.stop()

# Item properties
ITEM_PROPS = {
    "pizza": {"lcu": 6, "hold_time": 2},
    "wings_2h": {"lcu": 5, "hold_time": 2},
    "wings_4h": {"lcu": 8, "hold_time": 4},
    "baked_goods": {"lcu": 1, "hold_time": 24},
}

features = {
    "decision_hour": hour,
    "store_type": store_type,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "num_oven_items": len(item_config),
}

for item, demand in item_config.items():
    props = ITEM_PROPS[item]
    features[f"{item}_forecast_demand"] = demand
    features[f"{item}_lcu"] = props["lcu"]
    features[f"{item}_hold_time"] = props["hold_time"]
    features[f"{item}_time_remaining"] = max(0.5, props["hold_time"] - 0.25)
    features[f"{item}_cooked_qty"] = demand

# --- Run all three approaches ---
st.markdown("---")

# Context summary
st.markdown(f"### Scenario: {day_of_week} at {HOUR_LABELS.get(hour, f'{hour}:00')}")
st.markdown(f"**{STORE_TYPE_LABELS[store_type]}** — {len(item_config)} items competing for oven")

# Items display
item_cols = st.columns(len(item_config))
for i, (item, demand) in enumerate(item_config.items()):
    with item_cols[i]:
        emoji = ITEM_EMOJIS.get(item, "")
        st.metric(
            f"{emoji} {ITEM_DISPLAY_NAMES[item]}",
            f"{demand} units",
            help=f"Forecast generated from time of day & store type. Hold time: {ITEM_PROPS[item]['hold_time']}hr, LCU: {ITEM_PROPS[item]['lcu']}",
        )
        st.caption("📈 forecast")

st.markdown("---")

# Get rankings
v1_ranking = get_v1_ranking(features)
associate_pick = get_associate_pick(features)
v22_ranking = get_v22_ranking(features, model, historical)

# Three-column display
st.markdown("### Recommended Cook Order")

col_a, col_v1, col_v22 = st.columns(3)

with col_a:
    st.markdown("#### 👤 Associate")
    st.caption("Typical behavior")
    emoji = ITEM_EMOJIS.get(associate_pick, "")
    st.markdown(f"**Would likely pick:** {emoji} {ITEM_DISPLAY_NAMES.get(associate_pick, associate_pick)}")
    st.markdown("*Based on habit/familiarity*")

with col_v1:
    st.markdown("#### 📐 Rule-Based (v1)")
    st.caption("urgency × demand formula")
    for i, item in enumerate(v1_ranking):
        emoji = ITEM_EMOJIS.get(item, "")
        marker = "🥇" if i == 0 else f"#{i+1}"
        st.markdown(f"{marker} {emoji} {ITEM_DISPLAY_NAMES.get(item, item)}")

with col_v22:
    st.markdown("#### 🤖 ML System (v2.2)")
    st.caption("Pattern-based recommendation")
    for i, item in enumerate(v22_ranking):
        emoji = ITEM_EMOJIS.get(item, "")
        marker = "🥇" if i == 0 else f"#{i+1}"
        st.markdown(f"{marker} {emoji} {ITEM_DISPLAY_NAMES.get(item, item)}")

# --- Explanation ---
st.markdown("---")
st.markdown("### Why this order?")

explanations = generate_explanation(v22_ranking, features)
for exp in explanations:
    st.markdown(f"- {exp}")

# --- What changes? ---
st.markdown("---")
st.markdown("### Try adjusting...")
st.markdown("""
- **Change the hour** to see how forecast demand shifts (pizza peaks at lunch, wings at dinner, baked goods in the morning)
- **Switch store type** to see how demand scales (urban: ×1.4, suburban: ×1.0, highway: ×0.7)
- **Toggle weekend** to see the weekend uplift (×1.3 across all items)
- **Toggle items** to see how the ranking adapts when fewer items compete for the oven
""")

# --- Confidence indicator ---
if len(v22_ranking) >= 2 and v22_ranking != v1_ranking:
    st.markdown("---")
    st.info(
        f"💡 **ML disagrees with the formula:** The ML system recommends "
        f"**{ITEM_DISPLAY_NAMES[v22_ranking[0]]}** first, while the rule-based "
        f"system would pick **{ITEM_DISPLAY_NAMES[v1_ranking[0]]}**. "
        f"This is where the ML's learned patterns provide extra value."
    )
elif len(v22_ranking) >= 2:
    st.success(
        "✅ **All systems agree:** Both the formula and the ML system "
        f"recommend **{ITEM_DISPLAY_NAMES[v22_ranking[0]]}** first."
    )
