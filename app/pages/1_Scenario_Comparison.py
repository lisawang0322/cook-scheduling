"""Page 1: Live Scenario Comparison

Pick a scenario and see Associate vs v1 vs v2.2 rankings side-by-side.
"""

import streamlit as st
import random
import sys
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils import (
    load_model, load_labeled_data, compare_scenario, generate_explanation,
    ITEM_DISPLAY_NAMES, ITEM_EMOJIS, STORE_TYPE_LABELS, HOUR_LABELS,
    STORY_SCENARIOS, find_story_scenario, OVEN_ITEMS,
)

st.set_page_config(page_title="Scenario Comparison", page_icon="📊", layout="wide")

st.title("📊 Scenario Comparison")
st.markdown("Pick a real scenario and see how three approaches compare.")

# --- Load data ---
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_data():
    return load_labeled_data()

model, historical = get_model()
labeled_data = get_data()

# --- Scenario Selection ---
st.sidebar.markdown("### Select Scenario")

selection_mode = st.sidebar.radio(
    "How to choose a scenario:",
    ["Story scenario", "Random from data", "Pick by index"],
)

scenario = None

if selection_mode == "Story scenario":
    story_names = [s["name"] for s in STORY_SCENARIOS]
    story_idx = st.sidebar.selectbox("Choose a story:", range(len(story_names)),
                                     format_func=lambda i: story_names[i])
    scenario = find_story_scenario(labeled_data, story_idx)
    if scenario:
        st.sidebar.info(STORY_SCENARIOS[story_idx]["description"])
    else:
        st.sidebar.warning("No matching scenario found. Try another story.")

elif selection_mode == "Random from data":
    if st.sidebar.button("🎲 Generate Random Scenario"):
        st.session_state["random_idx"] = random.randint(0, len(labeled_data) - 1)
    idx = st.session_state.get("random_idx", 0)
    scenario = labeled_data[idx]

elif selection_mode == "Pick by index":
    idx = st.sidebar.number_input("Scenario index:", 0, len(labeled_data) - 1, 0)
    scenario = labeled_data[idx]

# --- Display comparison ---
if scenario:
    features = scenario["features"]
    result = compare_scenario(scenario, model, historical)

    # Context card
    st.markdown("---")
    st.markdown("### Scenario Context")

    ctx_cols = st.columns(4)
    ctx_cols[0].markdown(f"**Store Type**\n\n{STORE_TYPE_LABELS.get(features['store_type'], features['store_type'])}")
    ctx_cols[1].markdown(f"**Day**\n\n{features['day_of_week']}")
    ctx_cols[2].markdown(f"**Hour**\n\n{HOUR_LABELS.get(features['decision_hour'], str(features['decision_hour']))}")
    ctx_cols[3].markdown(f"**Items Present**\n\n{features.get('num_oven_items', len(result['v22_ranking']))}")

    # Items present with demand
    st.markdown("#### Items at this decision point:")
    item_cols = st.columns(4)
    col_idx = 0
    for item in OVEN_ITEMS:
        if f"{item}_forecast_demand" in features:
            with item_cols[col_idx % 4]:
                demand = features[f"{item}_forecast_demand"]
                time_rem = features[f"{item}_time_remaining"]
                emoji = ITEM_EMOJIS.get(item, "")
                st.markdown(
                    f"**{emoji} {ITEM_DISPLAY_NAMES[item]}**\n\n"
                    f"Demand: {demand} units\n\n"
                    f"Time left: {time_rem:.1f} hrs"
                )
            col_idx += 1

    # Three-column comparison
    st.markdown("---")
    st.markdown("### Cook Order Recommendations")

    col_a, col_v1, col_v22 = st.columns(3)

    with col_a:
        st.markdown("#### 👤 Associate")
        st.caption("What typically happens today")
        pick = result["associate_pick"]
        emoji = ITEM_EMOJIS.get(pick, "")
        st.markdown(f"**First pick:** {emoji} {ITEM_DISPLAY_NAMES.get(pick, pick)}")
        if result["associate_correct"]:
            st.success("✅ Matches optimal")
        else:
            st.error("❌ Not optimal")
        st.markdown(
            "*Based on habit (pizza first), expiration checking, "
            "or random choice.*"
        )

    with col_v1:
        st.markdown("#### 📐 Rule-Based (v1)")
        st.caption("Simple formula: urgency × demand")
        for i, item in enumerate(result["v1_ranking"]):
            emoji = ITEM_EMOJIS.get(item, "")
            prefix = "🥇" if i == 0 else f"#{i+1}"
            st.markdown(f"{prefix} {emoji} {ITEM_DISPLAY_NAMES.get(item, item)}")
        if result["v1_correct"]:
            st.success("✅ Top pick matches optimal")
        else:
            st.error("❌ Top pick not optimal")

    with col_v22:
        st.markdown("#### 🤖 ML System (v2.2)")
        st.caption("Learns from patterns in data")
        for i, item in enumerate(result["v22_ranking"]):
            emoji = ITEM_EMOJIS.get(item, "")
            prefix = "🥇" if i == 0 else f"#{i+1}"
            st.markdown(f"{prefix} {emoji} {ITEM_DISPLAY_NAMES.get(item, item)}")
        if result["v22_correct"]:
            st.success("✅ Top pick matches optimal")
        else:
            st.error("❌ Top pick not optimal")

    # Explanation panel
    st.markdown("---")
    st.markdown("### Why this order? (ML System reasoning)")

    explanations = generate_explanation(result["v22_ranking"], features)
    for exp in explanations:
        st.markdown(f"- {exp}")

    # Optimal answer
    st.markdown("---")
    with st.expander("📋 What was actually optimal? (based on real waste outcomes)"):
        st.markdown(f"**Optimal first item:** {ITEM_EMOJIS.get(result['optimal_first'], '')} "
                    f"{ITEM_DISPLAY_NAMES.get(result['optimal_first'], result['optimal_first'])}")
        st.markdown(f"**Full optimal order:** {' > '.join(ITEM_DISPLAY_NAMES.get(i, i) for i in result['optimal_order'])}")
        st.caption("Determined by which ordering would have produced the least waste based on actual sales data.")

else:
    st.info("👈 Select a scenario from the sidebar to get started.")
