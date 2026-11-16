
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data import generate_transactions
from src.model import (
    get_model_bundle,
    predict_transaction,
    get_local_explanation
)
from src.llm import generate_llm_explanation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Intelligence | Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME AND CSS
# ============================================================

st.markdown("""
<style>
:root {
    --ink:#e8eef8;
    --muted:#9aabc3;
    --panel:#111d30;
    --line:#24354d;
}

.stApp {
    background: radial-gradient(
        circle at 10% 0%,
        #172943 0,
        #0b1422 42%,
        #09111d 100%
    );
    color:var(--ink);
}

[data-testid="stSidebar"] {
    background:#0b1727;
    border-right:1px solid #24354d;
}

.block-container {
    padding-top:1.6rem;
    padding-bottom:3rem;
    max-width:1500px;
}

h1,h2,h3 {
    color:#f4f7fc;
    letter-spacing:-.02em;
}

p,li {
    color:#c6d2e2;
}

div[data-testid="stMetric"] {
    background:linear-gradient(145deg,#14243a,#101c2e);
    border:1px solid #2a3d57;
    padding:16px 18px;
    border-radius:15px;
}

div[data-testid="stMetric"] label {
    color:#9fb1c9 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#f3f7ff;
}

.panel {
    background:linear-gradient(
        145deg,
        rgba(21,37,59,.95),
        rgba(13,25,42,.95)
    );
    border:1px solid #263a55;
    border-radius:16px;
    padding:18px 20px;
    margin-bottom:12px;
}

.eyebrow {
    color:#6fd6c2;
    font-size:.76rem;
    text-transform:uppercase;
    letter-spacing:.13em;
    font-weight:700;
}

.muted {
    color:#9aabc3;
}

.small-note {
    color:#a9bad0;
    font-size:.88rem;
}

.pill {
    display:inline-block;
    padding:5px 10px;
    border-radius:99px;
    font-size:.78rem;
    font-weight:700;
    background:#203a55;
    color:#b9e8df;
}

div.stButton > button {
    border-radius:10px;
    border:1px solid #3a5879;
    font-weight:650;
}

[data-testid="stTabs"] button {
    color:#c4d3e6;
}

hr {
    border-color:#263a55;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_bundle():
    return get_model_bundle()


@st.cache_data
def load_raw_data(n=5000):
    return generate_transactions(n=n, seed=42)


# Load trained model
bundle = load_bundle()

model = bundle["model"]
metrics = bundle["metrics"]

# Load synthetic transaction data
df = load_raw_data(5000)


# ============================================================
# GENERATE MODEL-BASED RISK SCORES
# ============================================================

def add_model_risk_columns(data, trained_model):

    data = data.copy()

    # Generate model-compatible features
    from src.model import encode_features

    X = encode_features(data)

    # Predict fraud probabilities
    probabilities = trained_model.predict_proba(X)[:, 1]

    data["risk_score"] = probabilities

    # Convert probability into risk categories
    data["risk_band"] = np.select(
        [
            data["risk_score"] >= 0.70,
            data["risk_score"] >= 0.35
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    return data


df = add_model_risk_columns(df, model)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(x):
    return f"₹{x:,.0f}"


def risk_color(label):

    return {
        "Low": "#22c55e",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }.get(label, "#94a3b8")


def section_intro(kicker, title, description):

    st.markdown(
        f'<div class="eyebrow">{kicker}</div>'
        f'<h2 style="margin-top:4px">{title}</h2>'
        f'<p class="muted">{description}</p>',
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ Fraud Intelligence")

    st.caption(
        "Ensemble learning · Risk prediction · LLM explanations"
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Analyze a transaction",
            "Transaction explorer",
            "Model lab",
            "How it works"
        ],
        index=0,
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**Environment**")

    st.markdown(
        '<span class="pill">DEMO MODE · SYNTHETIC DATA</span>',
        unsafe_allow_html=True
    )

    st.caption(f"{len(df):,} generated transactions")

    st.caption(
        "No real banking or customer data is used."
    )

    st.divider()

    st.caption(
        "Built for learning and demonstration. "
        "Not a production fraud decision system."
    )


# ============================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="eyebrow">RISK OPERATIONS CENTER</div>',
        unsafe_allow_html=True
    )

    st.title("Fraud Intelligence Dashboard")

    st.markdown(
        "A clear view of transaction activity, model-detected "
        "risk, and the signals behind suspicious behavior."
    )

    total = len(df)

    frauds = int(df["is_fraud"].sum())

    fraud_rate = frauds / total * 100 if total else 0

    avg_amount = df["amount"].mean()

    high_risk = int(
        (df["risk_band"] == "High").sum()
    )

    # Dashboard metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions monitored",
        f"{total:,}"
    )

    c2.metric(
        "Known fraud cases",
        f"{frauds:,}",
        f"{fraud_rate:.1f}% of transactions"
    )

    c3.metric(
        "Average transaction",
        money(avg_amount)
    )

    c4.metric(
        "High-risk flagged",
        f"{high_risk:,}"
    )

    st.markdown("")

    # Transaction activity and risk distribution
    left, right = st.columns([1.15, 0.85])

    with left:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.subheader("Transaction activity")

        daily = df.groupby(
            "day",
            as_index=False
        ).agg(
            transactions=("transaction_id", "count"),
            fraud_cases=("is_fraud", "sum")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily["day"],
                y=daily["transactions"],
                name="Transactions",
                mode="lines",
                line=dict(width=2, color="#62a8ff")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily["day"],
                y=daily["fraud_cases"],
                name="Known fraud",
                mode="lines",
                line=dict(width=2, color="#ff7777"),
                yaxis="y2"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(title="Transactions"),
            yaxis2=dict(
                title="Fraud cases",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Blue shows daily transaction activity. "
            "Red shows records carrying a known synthetic fraud label."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.subheader("Risk distribution")

        band = (
            df["risk_band"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .reset_index()
        )

        band.columns = [
            "Risk band",
            "Transactions"
        ]

        fig = px.pie(
            band,
            names="Risk band",
            values="Transactions",
            hole=0.62,
            color="Risk band",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#f59e0b",
                "High": "#ef4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=5, r=5, t=10, b=5),
            legend=dict(orientation="h", y=-0.08)
        )

        fig.update_traces(
            textinfo="percent"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Risk categories are based on the model's predicted "
            "fraud probabilities."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Channel and high-risk records
    a, b = st.columns(2)

    with a:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.subheader("Where risk signals appear")

        channel = df.groupby(
            "channel",
            as_index=False
        ).agg(
            transactions=("transaction_id", "count"),
            fraud_rate=("is_fraud", "mean")
        )

        channel["fraud_rate"] *= 100

        fig = px.bar(
            channel,
            x="channel",
            y="fraud_rate",
            color="channel",
            text_auto=".1f",
            color_discrete_sequence=[
                "#5aa9e6",
                "#6dd3b0",
                "#a78bfa"
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            showlegend=False,
            yaxis_title="Known fraud rate (%)",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Comparison of known synthetic fraud-label rates "
            "across transaction channels."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with b:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.subheader("Recent high-risk activity")

        recent = (
            df[df["risk_band"] == "High"]
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(6)
        )

        st.dataframe(
            recent[
                [
                    "transaction_id",
                    "amount",
                    "channel",
                    "risk_score",
                    "risk_band"
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn(
                    "Amount (₹)",
                    format="₹%.0f"
                ),
                "risk_score": st.column_config.ProgressColumn(
                    "Risk score",
                    min_value=0,
                    max_value=1,
                    format="%.0f%%"
                )
            }
        )

        st.caption(
            "These are the highest model-scored records "
            "in the synthetic demonstration dataset."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "**Important distinction:** Known fraud is the synthetic "
        "ground-truth label. Risk band is the model's prediction. "
        "They are not the same thing."
    )


# ============================================================
# PAGE 2: ANALYZE A TRANSACTION
# ============================================================

elif page == "Analyze a transaction":

    section_intro(
        "INDIVIDUAL DECISION SUPPORT",
        "Analyze a transaction",
        "Enter transaction details to estimate fraud risk. "
        "The model returns a probability, risk band, "
        "contributing signals, and a plain-language explanation."
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader("Transaction details")

    with st.form("transaction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            amount = st.number_input(
                "Transaction amount (₹)",
                min_value=1.0,
                max_value=1000000.0,
                value=12500.0,
                step=500.0
            )

            hour = st.slider(
                "Hour of transaction",
                0, 23, 22
            )

            account_age = st.number_input(
                "Account age (days)",
                min_value=0,
                max_value=10000,
                value=45
            )

        with col2:

            velocity = st.slider(
                "Transactions in the last hour",
                0, 20, 6
            )

            failed = st.slider(
                "Recent failed attempts",
                0, 10, 2
            )

            distance = st.number_input(
                "Distance from usual location (km)",
                min_value=0.0,
                max_value=20000.0,
                value=850.0,
                step=50.0
            )

        with col3:

            channel = st.selectbox(
                "Channel",
                ["Mobile", "Web", "POS"]
            )

            international = st.selectbox(
                "International transaction?",
                ["No", "Yes"]
            )

            merchant_risk = st.select_slider(
                "Merchant risk level",
                options=["Low", "Medium", "High"],
                value="Medium"
            )

            device_trust = st.select_slider(
                "Device trust",
                options=["Trusted", "Unknown", "Untrusted"],
                value="Unknown"
            )

        submitted = st.form_submit_button(
            "🛡️ Assess fraud risk",
            type="primary",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:

        payload = {
            "amount": amount,
            "hour": hour,
            "account_age_days": account_age,
            "txn_velocity_1h": velocity,
            "failed_attempts": failed,
            "distance_km": distance,
            "channel": channel,
            "international": 1 if international == "Yes" else 0,
            "merchant_risk": {
                "Low": 0,
                "Medium": 1,
                "High": 2
            }[merchant_risk],
            "device_trust": {
                "Trusted": 2,
                "Unknown": 1,
                "Untrusted": 0
            }[device_trust]
        }

        result = predict_transaction(
            model,
            payload
        )

        st.session_state["last_analysis"] = {
            "payload": payload,
            "result": result
        }

    if "last_analysis" in st.session_state:

        payload = st.session_state["last_analysis"]["payload"]

        result = st.session_state["last_analysis"]["result"]

        st.markdown("---")

        st.subheader("Assessment result")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Predicted fraud probability",
            f'{result["risk_score"]:.1%}'
        )

        r2.metric(
            "Risk category",
            result["risk_band"]
        )

        r3.metric(
            "Suggested next step",
            result["recommendation"]
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["risk_score"] * 100,
                number={
                    "suffix": "%",
                    "font": {
                        "color": "white",
                        "size": 34
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#9fb1c9"
                    },
                    "bar": {
                        "color": risk_color(result["risk_band"])
                    },
                    "bgcolor": "#172943",
                    "bordercolor": "#314761",
                    "steps": [
                        {
                            "range": [0, 35],
                            "color": "#173d37"
                        },
                        {
                            "range": [35, 70],
                            "color": "#59451d"
                        },
                        {
                            "range": [70, 100],
                            "color": "#542b35"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=245,
            margin=dict(l=20, r=20, t=25, b=5)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "#### Why did the model assign this risk?"
        )

        reasons = get_local_explanation(
            payload,
            result
        )

        for reason in reasons:
            st.markdown(f"- {reason}")

        with st.expander(
            "View the submitted transaction features"
        ):
            st.json(payload)

        st.markdown("#### AI explanation")

        st.caption(
            "The LLM converts model outputs and risk signals "
            "into readable analyst notes. It does not make the prediction."
        )

        if st.button(
            "✨ Generate explanation with Gemini",
            key="gemini_explain"
        ):

            with st.spinner("Preparing explanation..."):

                explanation, source = generate_llm_explanation(
                    payload,
                    result,
                    reasons
                )

            st.markdown(
                '<div class="panel">'
                + str(explanation)
                + '</div>',
                unsafe_allow_html=True
            )

            st.caption(source)

        if not os.getenv("GEMINI_API_KEY"):

            st.caption(
                "No Gemini API key detected. Add GEMINI_API_KEY "
                "to .env to enable LLM-generated explanations. "
                "The local explanation remains available."
            )

    else:

        st.info(
            "Complete the form and click "
            "**Assess fraud risk** to see an assessment."
        )


# ============================================================
# PAGE 3: TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction explorer":

    section_intro(
        "INVESTIGATION WORKSPACE",
        "Transaction explorer",
        "Filter the synthetic transaction ledger, inspect "
        "individual records, and compare predicted risk "
        "with known demo labels."
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        bands = st.multiselect(
            "Risk band",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

    with f2:

        channels = st.multiselect(
            "Channel",
            sorted(df["channel"].unique()),
            default=sorted(df["channel"].unique())
        )

    with f3:

        search = st.text_input(
            "Search transaction ID",
            placeholder="e.g. TXN-00125"
        )

    min_amt = int(df["amount"].min())

    max_amt = int(df["amount"].max())

    if min_amt == max_amt:
        amount_range = (min_amt, max_amt)
    else:
        amount_range = st.slider(
            "Transaction amount range (₹)",
            min_amt,
            max_amt,
            (min_amt, max_amt),
            step=100
        )

    filtered = df[
        df["risk_band"].isin(bands)
        & df["channel"].isin(channels)
        & df["amount"].between(*amount_range)
    ]

    if search.strip():

        filtered = filtered[
            filtered["transaction_id"].astype(str).str.contains(
                search.strip(),
                case=False,
                regex=False
            )
        ]

    st.markdown(
        f"**{len(filtered):,} matching transactions**"
    )

    st.dataframe(
        filtered[
            [
                "transaction_id",
                "day",
                "amount",
                "channel",
                "risk_score",
                "risk_band",
                "is_fraud",
                "hour",
                "txn_velocity_1h",
                "failed_attempts"
            ]
        ].sort_values(
            "risk_score",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
        height=390,
        column_config={
            "amount": st.column_config.NumberColumn(
                "Amount (₹)",
                format="₹%.0f"
            ),
            "risk_score": st.column_config.ProgressColumn(
                "Model risk",
                min_value=0,
                max_value=1,
                format="%.0f%%"
            ),
            "is_fraud": st.column_config.CheckboxColumn(
                "Known fraud"
            )
        }
    )

    st.markdown("#### Inspect one transaction")

    ids = filtered["transaction_id"].tolist()

    if ids:

        selected = st.selectbox(
            "Choose transaction",
            ids
        )

        record = filtered[
            filtered["transaction_id"] == selected
        ].iloc[0]

        left, right = st.columns(2)

        with left:

            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True
            )

            st.write(f"**{record.transaction_id}**")

            st.metric(
                "Amount",
                money(record.amount)
            )

            st.write(
                f"Channel: **{record.channel}**"
            )

            st.write(
                f"Model risk: **{record.risk_band} "
                f"({record.risk_score:.1%})**"
            )

            st.write(
                "Known demo label: **"
                + ("Fraud" if record.is_fraud else "Legitimate")
                + "**"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with right:

            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True
            )

            st.write("**Analyst interpretation**")

            if record.risk_band == "High":

                st.write(
                    "The model estimates elevated risk. "
                    "Review the transaction context and verify "
                    "the customer before taking action."
                )

            elif record.risk_band == "Medium":

                st.write(
                    "The model sees mixed signals. "
                    "Additional checks may help resolve uncertainty."
                )

            else:

                st.write(
                    "The model estimates lower risk based on "
                    "the available input signals. Low risk "
                    "does not guarantee legitimacy."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "No transactions match these filters. "
            "Try widening the filters."
        )


# ============================================================
# PAGE 4: MODEL LAB
# ============================================================

elif page == "Model lab":

    section_intro(
        "MODEL PERFORMANCE & TRANSPARENCY",
        "Ensemble model lab",
        "Understand what the model learned, how it was "
        "evaluated, and why combining models can improve robustness."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "ROC-AUC",
        f'{metrics["roc_auc"]:.3f}'
    )

    b.metric(
        "Precision",
        f'{metrics["precision"]:.3f}'
    )

    c.metric(
        "Recall",
        f'{metrics["recall"]:.3f}'
    )

    d.metric(
        "F1 score",
        f'{metrics["f1"]:.3f}'
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader("What do these metrics mean?")

    st.markdown("""
    - **Precision:** When the system raises an alert, how often is it right?
    - **Recall:** How much of the actual fraud did the system catch?
    - **F1:** Combines precision and recall into one score.
    - **ROC-AUC:** Measures ranking quality across different thresholds.

    Fraud detection often has an imbalance: legitimate transactions
    are much more common than fraud. Accuracy alone can therefore
    be misleading.
    """)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.subheader("Confusion matrix")

        cm = metrics["confusion_matrix"]

        fig = px.imshow(
            cm,
            text_auto=True,
            x=[
                "Predicted legitimate",
                "Predicted fraud"
            ],
            y=[
                "Actual legitimate",
                "Actual fraud"
            ],
            color_continuous_scale="Blues",
            aspect="auto"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=330,
            margin=dict(l=5, r=5, t=10, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Diagonal cells show correct predictions. "
            "Off-diagonal cells show mistakes."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.subheader("Most influential features")

        imp = (
            bundle["feature_importance"]
            .head(10)
            .sort_values("importance")
        )

        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Teal"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=330,
            margin=dict(l=5, r=5, t=10, b=5),
            xaxis_title="Relative importance",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Feature importance is a global summary. "
            "It does not prove that a feature caused a specific prediction."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader("Why use ensemble learning?")

    st.markdown("""
    This project combines **Random Forest** and
    **Gradient Boosting** using soft voting:

    1. Random Forest trains many decision trees on varied samples
       and combines their predictions.
    2. Gradient Boosting builds trees sequentially, focusing
       on patterns earlier trees missed.
    3. Soft voting averages the predicted class probabilities
       from both models.

    The aim is to combine different learning strategies rather
    than rely on a single estimator. The displayed results are
    from a held-out synthetic test split, not a real-world benchmark.
    """)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 5: HOW IT WORKS
# ============================================================

elif page == "How it works":

    section_intro(
        "PROJECT GUIDE",
        "How the system works",
        "A plain-language walkthrough you can use to understand "
        "the project and explain it in a technical interview."
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.subheader("1. Problem statement")

    st.write(
        "Banks and digital payment platforms process many "
        "transactions. A small portion may be fraudulent. "
        "Reviewing every transaction manually is slow, while "
        "blocking every unusual payment frustrates genuine customers. "
        "The goal is to estimate transaction risk and help an "
        "analyst prioritize reviews."
    )

    st.subheader("2. Proposed solution")

    st.write(
        "The system takes transaction attributes, passes them "
        "to an ensemble machine-learning model, estimates the "
        "probability of fraud, assigns a Low / Medium / High "
        "risk band, and explains the signals. An optional LLM "
        "converts those signals into a readable analyst summary."
    )

    st.subheader("3. End-to-end workflow")

    st.markdown("""
    **Transaction details → Data preparation → Ensemble model
    → Fraud probability → Risk band → Explanation → Analyst review**

    - **Input:** amount, time, account age, transaction velocity,
      failed attempts, location distance, channel, merchant risk,
      and device trust.
    - **Prediction:** the ensemble estimates a probability between 0 and 1.
    - **Decision support:** probability is grouped into Low,
      Medium, or High risk.
    - **Explanation:** rule-based signals explain the model output;
      Gemini can optionally turn them into a natural-language note.
    """)

    st.subheader("4. Technologies used")

    st.markdown("""
    | Technology | Purpose |
    |---|---|
    | Python | Core programming language |
    | Streamlit | Interactive dashboard and user interface |
    | Pandas / NumPy | Data generation and processing |
    | Scikit-learn | Training and evaluating ML models |
    | Random Forest + Gradient Boosting | Ensemble fraud prediction |
    | Plotly | Interactive charts |
    | Gemini API (optional) | Natural-language explanations |
    """)

    st.subheader("5. Important limitations")

    st.warning(
        "This is an educational prototype using synthetic data. "
        "It is not trained on real bank transactions, is not "
        "calibrated for real financial decisions, and must not "
        "be used to approve, reject, or block actual payments. "
        "A production system would need secure data pipelines, "
        "privacy controls, drift monitoring, human review, "
        "fairness testing, and rigorous validation."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Fraud Intelligence · Synthetic demonstration · "
    "Model predictions are decision-support signals, "
    "not proof of fraud."
)
