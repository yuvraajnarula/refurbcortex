import streamlit as st
import requests
import base64
import pandas as pd

API_URL = "http://localhost:8000/api/v1/predict"

st.set_page_config(
    page_title="RefurbCortex AI",
    layout="wide"
)

st.title("RefurbCortex AI | Metacognitive Inspection Engine")
st.caption("System 1 Vision -> Confidence Routing -> Human-in-the-Loop Fallback")

# Initialize session state for closed-loop feedback
if "last_inspection" not in st.session_state:
    st.session_state.last_inspection = None

with st.sidebar:
    st.header("Vehicle & Damage Context")

    brand = st.text_input("Brand", "Maruti Suzuki")
    model = st.text_input("Model", "Swift")

    year = st.number_input(
        "Year",
        min_value=2010,
        max_value=2024,
        value=2018
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["petrol", "diesel", "ev", "cng"]
    )

    panel = st.selectbox(
        "Panel Affected",
        [
            "side_mirror",
            "roof",
            "front_bumper",
            "left_door",
            "windshield",
            "hood"
        ]
    )

    category = st.selectbox(
        "Damage Category",
        [
            "scratch",
            "dent",
            "corrosion",
            "electrical",
            "structural"
        ]
    )

    city = st.text_input("City", "Hyderabad")

    tier = st.selectbox(
        "City Tier",
        ["tier_1", "tier_2", "tier_3"]
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )
    is_ev = fuel == "ev"
    if is_ev:
        st.divider()
        st.subheader("EV Diagnostics")
        fast_charge = st.slider("Fast Charge %", 0, 100, 30)
        avg_temp = st.slider("Avg Climate °C", 15, 45, 30)
        odometer = st.number_input("Odometer (km)", 0, 200000, 50000)

    run_btn = st.button(
        "Run Inspection",
        type="primary",
        disabled=uploaded is None
    )

if uploaded is not None and run_btn:

    st.image(
        uploaded,
        caption="Original Image",
        use_container_width=True
    )

    with st.spinner("Analyzing damage and evaluating confidence..."):

        files = {
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type
            )
        }

        data = {
            "vehicle_brand": brand,
            "vehicle_model": model,
            "manufacture_year": year,
            "fuel_type": fuel,
            "panel_affected": panel,
            "damage_category": category,
            "city": city,
            "city_tier": tier,
            "is_ev": is_ev,
            "odometer_km": odometer if is_ev else 50000
        }

        try:
            response = requests.post(
                API_URL,
                files=files,
                data=data,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()

            # Capture successful run for feedback
            st.session_state.last_inspection = {
                "id": result["inspection_id"],
                "predicted": result["summary"].get("total_adjusted_cost_inr", 0)
            }

            # Heatmap
            if result.get("heatmap_b64"):
                heatmap_bytes = base64.b64decode(
                    result["heatmap_b64"]
                )

                st.image(
                    heatmap_bytes,
                    caption="AI Damage Heatmap",
                    use_container_width=True
                )
            if is_ev and result.get("recommendation"):
                st.divider()
                st.subheader("🔋 EV Battery Health Impact")
                ev_note = result["recommendation"]["reasoning"].split("| EV")[1].split("|")[0] if "| EV" in result["recommendation"]["reasoning"] else "N/A"
                st.info(f"{ev_note.strip()}")
            # Routing
            route = result.get("routing", {})

            st.markdown(
                f"### Routing Status: `{route.get('status', 'UNKNOWN')}`"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                confidence = float(
                    route.get("confidence", 0)
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )

            with col2:
                total_cost = (
                    result.get("summary", {})
                    .get("total_adjusted_cost_inr", 0)
                )

                st.metric(
                    "Adjusted Refurb Cost",
                    f"₹{total_cost:,.0f}"
                )

            with col3:
                inference_time = (
                    result.get("meta", {})
                    .get("inference_time_ms", 0)
                )

                st.metric(
                    "Inference Time",
                    f"{inference_time} ms"
                )

            if route.get("requires_human_review", False):
                st.warning(
                    "Low confidence detected. "
                    "Applied historical safety margin. "
                    "Recommended: request better lighting or angle "
                    "or escalate to a human grader."
                )
            else:
                st.success(
                    "High confidence. Proceeding to trade-off "
                    "analysis and cost optimization."
                )

            # Damage Breakdown
            damage_records = result.get(
                "damage_records",
                []
            )

            if damage_records:

                st.subheader(
                    "Damage & Cost Breakdown"
                )

                df = pd.DataFrame(
                    damage_records
                )

                required_columns = [
                    "panel_affected",
                    "damage_category",
                    "severity_label",
                    "repair_cost_min_inr",
                    "repair_cost_max_inr",
                    "ai_confidence",
                    "recommended_action",
                    "repair_priority"
                ]

                available_columns = [
                    col
                    for col in required_columns
                    if col in df.columns
                ]

                st.dataframe(
                    df[available_columns].style.format(
                        {
                            "repair_cost_min_inr": "₹{:,.0f}",
                            "repair_cost_max_inr": "₹{:,.0f}",
                            "ai_confidence": "{:.2%}"
                        }
                    ),
                    use_container_width=True
                )

                st.subheader(
                    "AI Recommendations"
                )

                for rec in damage_records:

                    panel_name = (
                        rec.get(
                            "panel_affected",
                            "Unknown Panel"
                        )
                        .replace("_", " ")
                        .title()
                    )

                    severity = rec.get(
                        "severity_label",
                        "Unknown"
                    )

                    min_cost = rec.get(
                        "repair_cost_min_inr",
                        0
                    )

                    max_cost = rec.get(
                        "repair_cost_max_inr",
                        0
                    )

                    expander_title = (
                        f"{panel_name} | "
                        f"{severity} | "
                        f"₹{min_cost:,.0f} - ₹{max_cost:,.0f}"
                    )

                    with st.expander(
                        expander_title
                    ):

                        st.write(
                            f"**Action:** "
                            f"{rec.get('recommended_action', 'N/A')}"
                        )

                        st.write(
                            f"**Priority:** "
                            f"{rec.get('repair_priority', 'N/A')}"
                        )

                        st.write(
                            f"**Safety Risk:** "
                            f"{str(rec.get('safety_risk', 'N/A')).upper()}"
                        )

                        st.write(
                            f"**Resale Impact:** "
                            f"₹{rec.get('predicted_resale_impact_inr', 0):,.0f}"
                        )

                        st.write(
                            f"**Expected Profit:** "
                            f"₹{rec.get('expected_profit_after_repair_inr', 0):,.0f}"
                        )

                        st.write(
                            f"**AI Confidence:** "
                            f"{rec.get('ai_confidence', 0) * 100:.1f}%"
                        )

            else:
                st.info(
                    "No significant damage detected in the current view."
                )

            # SHAP XAI Breakdown
            if result.get("shap_breakdown"):
                st.divider()
                st.subheader("Explainable AI: Cost Attribution Breakdown")
                shap = result["shap_breakdown"]
                
                col_x1, col_x2 = st.columns(2)
                col_x1.metric("Base Estimate", f"₹{shap['base_estimate_inr']:,.0f}")
                col_x2.metric("XAI Adjusted", f"₹{shap['xai_adjusted_cost_inr']:,.0f}")
                
                # Horizontal bar chart of top drivers
                drivers_df = pd.DataFrame(shap["top_drivers"])
                drivers_df["Contribution (₹)"] = drivers_df["contribution_inr"]
                drivers_df.set_index("feature", inplace=True)
                
                # Use st.bar_chart (vertical) - simple; for horizontal we'd need altair/vega
                # Present as vertical bar chart for simplicity
                st.bar_chart(drivers_df[["Contribution (₹)"]])
                
                st.info(
                    "Interpretation: SHAP values show how each feature pushes the estimate up/down vs the baseline. "
                    "Higher Severity Score & Labor Hours drive positive cost adjustments."
                )

            # System 2: Agentic Trade-Off Analysis
            recommendation = result.get(
                "recommendation"
            )

            if recommendation:

                st.divider()

                st.subheader(
                    "System 2: Agentic Trade-Off Analysis"
                )

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric(
                        "Recommendation",
                        recommendation.get(
                            "recommendation",
                            "N/A"
                        )
                    )

                with col_b:
                    st.metric(
                        "Net Margin",
                        f"{recommendation.get('expected_net_margin_pct', 0)}%"
                    )

                with col_c:
                    st.metric(
                        "Turnover Risk",
                        recommendation.get(
                            "turnover_risk",
                            "N/A"
                        )
                    )

                st.info(
                    f"Reasoning: "
                    f"{recommendation.get('reasoning', 'No reasoning provided.')}"
                )

                priority_repairs = recommendation.get(
                    "repair_priority_items",
                    []
                )

                if priority_repairs:
                    st.write(
                        "**Priority Repairs:**"
                    )

                    st.write(
                        " • ".join(
                            priority_repairs
                        )
                    )

            else:
                if route.get(
                    "requires_human_review",
                    False
                ):
                    st.warning(
                        "System 2 skipped: "
                        "Confidence below threshold. "
                        "Route to human grader first."
                    )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to backend. "
                "Ensure FastAPI is running on "
                "http://localhost:8000"
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. "
                "The backend may be overloaded "
                "or unavailable."
            )

        except requests.exceptions.HTTPError:
            st.error(
                f"API Error ({response.status_code}): "
                f"{response.text}"
            )

        except Exception as e:
            st.error(
                f"Unexpected error: {str(e)}"
            )

# --- Closed-Loop Feedback Form ---
if st.session_state.last_inspection:
    st.divider()
    st.subheader("Closed-Loop Feedback: Submit Actual Repair Cost")
    st.caption("This updates ChromaDB memory & calibrates future confidence routing.")
    
    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        actual_cost = col1.number_input("Actual Repair Cost (₹)", min_value=0, step=500, value=0)
        override = col2.checkbox("Human Grader Override?")
        notes = st.text_area("Grader Notes (optional)", placeholder="e.g., Hidden rust, parts backordered, customer waived repair")
        submit_fb = st.form_submit_button("Submit Feedback")
        
        if submit_fb:
            if actual_cost <= 0:
                st.warning("Please enter a valid actual cost.")
            else:
                with st.spinner("Logging feedback to memory..."):
                    fb_payload = {
                        "inspection_id": st.session_state.last_inspection["id"],
                        "predicted_cost_inr": st.session_state.last_inspection["predicted"],
                        "actual_cost_inr": actual_cost,
                        "human_override": override,
                        "notes": notes
                    }
                    try:
                        fb_resp = requests.post(f"{API_URL.replace('/predict', '/feedback')}", json=fb_payload)
                        if fb_resp.status_code == 200:
                            fb_data = fb_resp.json()["data"]
                            st.success(f"Feedback Logged! Error: {fb_data['error_rate_pct']}% | ChromaDB updated.")
                            st.session_state.last_inspection = None
                        else:
                            st.error(f"API Error: {fb_resp.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")