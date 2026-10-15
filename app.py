import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import xgboost as xgb


# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Credit Risk Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium fintech design
st.markdown("""
<style>
    :root {
        --primary-bg: #0a192f;
        --secondary-bg: #112240;
        --accent-color: #64b5f6;
        --accent-hover: #42a5f5;
        --text-primary: #e6f1ff;
        --text-secondary: #94a3b8;
        --border-color: #233554;
        --success-color: #4db6ac;
        --warning-color: #ffb74d;
        --danger-color: #ef5350;
    }
    
    body {
        background-color: var(--primary-bg);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-primary);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg);
    }
    
    [data-testid="stSidebar"] .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid var(--border-color);
        transition: border-color 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent-color);
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    
    /* Main Content */
    .main-header {
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        color: var(--text-primary);
        line-height: 1.2;
        letter-spacing: 0.3px;
    }
    
    .main-header p {
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Cards */
    .stCard {
        background-color: var(--secondary-bg);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stCard:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--secondary-bg);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--accent-color);
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 600;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--success-color);
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--accent-color);
        color: var(--primary-bg);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: var(--accent-hover);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(100, 181, 246, 0.3);
    }
    
    .stButton button[type="primary"] {
        background-color: var(--accent-color);
        color: var(--primary-bg);
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .stButton button[type="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(100, 181, 246, 0.3);
    }
    
    .stDownloadButton button {
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton button:hover {
        background-color: var(--accent-hover);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(100, 181, 246, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--accent-color);
        box-shadow: 0 2px 8px rgba(100, 181, 246, 0.15);
    }
    
    .streamlit-expanderContent {
        background-color: var(--primary-bg);
        border: 1px solid var(--border-color);
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* Table */
    .dataframe {
        background-color: var(--secondary-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .dataframe th {
        background-color: #1a2d4d;
        color: var(--text-primary);
        font-weight: 600;
        padding: 0.75rem;
    }
    
    .dataframe td {
        color: var(--text-secondary);
        padding: 0.5rem 0.75rem;
    }
    
    /* Progress Bar */
    .stProgress .progress-bar {
        background-color: var(--accent-color);
    }
    
    /* Success/Warning/Error Boxes */
    .stSuccess {
        background-color: rgba(77, 182, 172, 0.15);
        border-left: 4px solid var(--success-color);
    }
    
    .stWarning {
        background-color: rgba(255, 183, 77, 0.15);
        border-left: 4px solid var(--warning-color);
    }
    
    .stError {
        background-color: rgba(239, 83, 80, 0.15);
        border-left: 4px solid var(--danger-color);
    }
    
    /* Section Headers */
    .section-header {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .spacing-sm { margin: 0.5rem 0; }
    .spacing-md { margin: 1rem 0; }
    .spacing-lg { margin: 1.5rem 0; }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stCard { padding: 1rem; }
        .matplotlib-container { padding: 0.5rem; max-width: 100%; }
        .main-header h1 { font-size: 1.5rem; }
        .section-header { font-size: 1.1rem; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.5rem; }
    }
    
    /* Sidebar Navigation */
    [data-testid="stSidebarNav"] {
        color: var(--text-primary);
    }
    
    [data-testid="stSidebarNav"] li {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebarNav"] a {
        color: var(--text-primary);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebarNav"] a:hover {
        background-color: var(--accent-color);
        color: var(--primary-bg);
    }
    
    [data-testid="stSidebarNav"] li.active a {
        background-color: var(--accent-color);
        color: var(--primary-bg);
    }
    
    /* Navigation item styling */
    [data-testid="stSidebarNav"] li a span {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }
    
    /* Chart Styling */
    .matplotlib-container {
        background-color: var(--secondary-bg);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .matplotlib-container .figure {
        background-color: var(--secondary-bg);
    }
</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent

PREPROCESSOR_PATH = BASE_DIR / "credit_risk_preprocessor.joblib"
NATIVE_MODEL_PATH = BASE_DIR / "credit_risk_xgb_native.json"
METADATA_PATH = BASE_DIR / "credit_risk_metadata.json"
DEFAULTS_PATH = BASE_DIR / "credit_risk_defaults.json"


# ============================================================
# 2. VERIFY REQUIRED FILES
# ============================================================

required_files = [
    PREPROCESSOR_PATH,
    NATIVE_MODEL_PATH,
    METADATA_PATH,
    DEFAULTS_PATH
]

missing_files = [
    file.name for file in required_files
    if not file.exists()
]

if missing_files:
    st.error(
        "Required project files are missing:\n\n"
        + "\n".join(f"- {name}" for name in missing_files)
    )
    st.info(
        "Place all four files in the same directory as app.py."
    )
    st.stop()


# ============================================================
# 3. LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_models():
    # Load the fitted preprocessing pipeline
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Load XGBoost using its native JSON format
    classifier = xgb.XGBClassifier()
    classifier.load_model(str(NATIVE_MODEL_PATH))

    return preprocessor, classifier


preprocessor, classifier = load_models()

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

with open(DEFAULTS_PATH, "r", encoding="utf-8") as file:
    defaults = json.load(file)

FEATURES = metadata["features"]
THRESHOLD = float(metadata["classification_threshold"])

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
REPAYMENT = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Check the expected input columns before making predictions
if set(FEATURES) - set(defaults):
    st.error(
        "The defaults JSON does not contain all model features. "
        "Regenerate it from your Colab training data."
    )
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def predict_default_probabilities(data):
    """
    Apply the saved training-time preprocessing and return
    the probability of DEFAULT_NEXT_MONTH = 1.
    """
    missing = [col for col in FEATURES if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    input_data = data[FEATURES].copy()
    transformed = preprocessor.transform(input_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    probabilities = classifier.predict_proba(transformed)[:, 1]

    return np.asarray(probabilities, dtype=float)


def score_customers(data):
    """Score an entire customer portfolio."""
    probabilities = predict_default_probabilities(data)

    result = data.copy()
    result["Default_Probability"] = probabilities
    result["Predicted_Default"] = (
        probabilities >= THRESHOLD
    ).astype(int)

    result["Risk_Category"] = np.where(
        probabilities >= THRESHOLD,
        "High",
        "Below Threshold"
    )

    return result


def format_currency(value):
    """Format value as currency."""
    return f"${value:,.0f}"


def get_risk_category(probability):
    """Get risk category based on probability."""
    if probability >= THRESHOLD:
        return "High", "#ef5350"
    elif probability >= THRESHOLD * 0.7:
        return "Elevated", "#ffb74d"
    else:
        return "Low", "#66bb6a"


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    # Brand Identity
    st.markdown(
        """
        <div style="padding: 1.5rem 0;">
            <h2 style="color: var(--text-primary); margin: 0; font-size: 1.5rem;">
                Credit<span style="color: var(--accent-color);">Risk</span>
            </h2>
            <p style="color: var(--text-secondary); margin: 0.25rem 0 0 0; font-size: 0.85rem;">
                Analytics Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "",
        [
            ("● Overview", "Overview"),
            ("› Customer Assessment", "Customer Assessment"),
            ("› Portfolio Scoring", "Portfolio Scoring"),
            ("› Model Insights", "Model Insights")
        ],
        format_func=lambda x: x[0],
        label_visibility="collapsed"
    )[1]
    
    st.divider()
    
    # Model Status
    st.markdown(
        """
        <div style="padding: 1rem; background-color: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid var(--border-color);">
            <h4 style="color: var(--text-primary); margin: 0 0 1rem 0; font-size: 0.9rem;">Model Status</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                <div>
                    <p style="color: var(--text-secondary); font-size: 0.75rem; margin: 0;">ROC-AUC</p>
                    <p style="color: var(--accent-color); font-weight: 600; margin: 0.25rem 0 0 0;">{:.4f}</p>
                </div>
                <div>
                    <p style="color: var(--text-secondary); font-size: 0.75rem; margin: 0;">Threshold</p>
                    <p style="color: var(--accent-color); font-weight: 600; margin: 0.25rem 0 0 0;">{:.2%}</p>
                </div>
            </div>
        </div>
        """.format(float(metadata.get('roc_auc', 0)), THRESHOLD),
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Footer
    st.markdown(
        """
        <div style="padding: 1rem; text-align: center;">
            <p style="color: var(--text-secondary); font-size: 0.75rem; margin: 0;">
                Educational prototype. <br>Not for real lending decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. MAIN CONTENT HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>{}</h1>
        <p>{}</p>
    </div>
    """.format(
        page,
        {
            "Overview": "Portfolio-level risk metrics and customer analytics",
            "Customer Assessment": "Evaluate individual credit default risk with comprehensive financial analysis",
            "Portfolio Scoring": "Batch score customer portfolios with automated risk classification",
            "Model Insights": "Model performance, feature importance, and explainability analysis"
        }[page]
    ),
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 7. OVERVIEW PAGE
# ============================================================

if page == "Overview":
    st.subheader("Portfolio Analytics")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info(
        "**Portfolio Overview**\n\n"
        "Upload a CSV containing the required model features to generate "
        "comprehensive portfolio risk metrics and visualizations."
    )
    
    uploaded = st.file_uploader(
        "Upload Customer Portfolio (CSV)",
        type=["csv"],
        key="overview_upload"
    )

    if uploaded is not None:
        try:
            portfolio = pd.read_csv(uploaded)

            missing = [
                col for col in FEATURES
                if col not in portfolio.columns
            ]

            if missing:
                st.error(
                    f"⚠️ The uploaded CSV is missing these required features: "
                    f"{', '.join(missing)}"
                )
                st.markdown(
                    "<p style='color: var(--text-secondary); font-size: 0.9rem;'>"
                    "Required features: " + ", ".join(FEATURES) +
                    "</p>",
                    unsafe_allow_html=True
                )

            elif portfolio.empty:
                st.warning("⚠️ The uploaded CSV contains no records.")

            else:
                scored = score_customers(portfolio)

                # Summary Metrics
                total_customers = len(scored)
                high_risk_count = int(scored["Predicted_Default"].sum())
                high_risk_pct = (high_risk_count / total_customers * 100) if total_customers > 0 else 0
                average_probability = scored["Default_Probability"].mean()

                st.markdown(
                    "<h3 class='section-header'>Portfolio Summary</h3>",
                    unsafe_allow_html=True
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Customers",
                    f"{total_customers:,}"
                )

                col2.metric(
                    "High Risk",
                    f"{high_risk_count:,}",
                    f"{high_risk_pct:.1f}% of portfolio"
                )

                col3.metric(
                    "Mean Probability",
                    f"{average_probability:.2%}"
                )

                col4.metric(
                    "Classification Threshold",
                    f"{THRESHOLD:.1%}"
                )

                # Observed outcomes
                if "DEFAULT_NEXT_MONTH" in portfolio.columns:
                    observed_rate = pd.to_numeric(
                        portfolio["DEFAULT_NEXT_MONTH"],
                        errors="coerce"
                    ).mean()
                    delta = observed_rate - average_probability

                    st.markdown("<br>", unsafe_allow_html=True)
                    col5, = st.columns(1)
                    col5.metric(
                        "Observed Default Rate",
                        f"{observed_rate:.2%}",
                        delta=f"{delta:+.2%}" if abs(delta) > 0.01 else None,
                        delta_color="inverse" if delta > 0 else "normal"
                    )

                # Visualizations
                st.markdown(
                    "<h3 class='section-header'>Risk Distribution</h3>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                left, right = st.columns(2)

                with left:
                    st.markdown("#### Risk Category Distribution")
                    st.markdown("<br>", unsafe_allow_html=True)

                    risk_counts = (
                        scored["Risk_Category"]
                        .value_counts()
                        .reindex(
                            ["High", "Below Threshold"],
                            fill_value=0
                        )
                    )

                    fig, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(
                        risk_counts.index,
                        risk_counts.values,
                        color=["#ef5350", "#66bb6a"],
                        edgecolor="none"
                    )
                    ax.set_ylabel("Number of Customers", color="#94a3b8")
                    ax.set_xlabel("Risk Category", color="#94a3b8")
                    ax.tick_params(colors="#94a3b8", labelsize=10)
                    ax.spines['bottom'].set_color('#233554')
                    ax.spines['left'].set_color('#233554')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    for bar, count in zip(bars, risk_counts.values):
                        ax.text(
                            bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 50,
                            f'{int(count)}',
                            ha='center',
                            va='bottom',
                            color='#94a3b8'
                        )

                    fig.patch.set_facecolor('#112240')
                    ax.set_facecolor('#112240')
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)

                with right:
                    st.markdown("#### Probability Distribution")
                    st.markdown("<br>", unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(8, 5))

                    ax.hist(
                        scored["Default_Probability"],
                        bins=30,
                        color="#64b5f6",
                        alpha=0.7,
                        edgecolor="none"
                    )

                    ax.axvline(
                        THRESHOLD,
                        linestyle="--",
                        linewidth=2,
                        color="#ffb74d",
                        label=f"Threshold: {THRESHOLD:.2f}"
                    )

                    ax.set_xlabel("Predicted Default Probability (probability)", color="#94a3b8")
                    ax.set_ylabel("Number of Customers", color="#94a3b8")
                    ax.set_title("Probability Distribution", color="#e6f1ff", fontweight="600")
                    ax.tick_params(colors="#94a3b8", labelsize=10)
                    ax.legend(facecolor='#1a2d4d', edgecolor='#233554', labelcolor='#94a3b8')

                    ax.spines['bottom'].set_color('#233554')
                    ax.spines['left'].set_color('#233554')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    fig.patch.set_facecolor('#112240')
                    ax.set_facecolor('#112240')
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)

                # High Risk Preview
                st.markdown(
                    "<h3 class='section-header'>High-Risk Customer Preview (Top 20)</h3>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                top_customers = scored.sort_values(
                    "Default_Probability",
                    ascending=False
                ).head(20).copy()

                top_customers["Risk_Level"] = top_customers["Default_Probability"].apply(
                    lambda x: "🔴 High" if x >= THRESHOLD else "🟡 Elevated"
                )

                preview_columns = ["LIMIT_BAL", "AGE", "BILL_UTILIZATION", "DELAYED_MONTHS_6M", "Default_Probability", "Risk_Level"]

                st.dataframe(
                    top_customers[preview_columns],
                    use_container_width=True,
                    hide_index=True
                )

                # Download Button
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download Scored Portfolio",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="credit_risk_scored_portfolio.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as exc:
            st.error(f"❌ Unable to score portfolio: {exc}")

    else:
        st.info(
            "**Upload a CSV file**\n\n"
            "Upload your cleaned customer CSV to calculate portfolio-level metrics. "
            "No customer results are fabricated."
        )


# ============================================================
# 8. INDIVIDUAL CUSTOMER ASSESSMENT
# ============================================================

elif page == "Customer Assessment":
    st.subheader("Individual Customer Risk Assessment")
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "**Customer Assessment Form**\n\n"
        "Enter the customer's financial information to generate a risk prediction. "
        "Initial values are default examples for testing."
    )

    with st.form("customer_form", clear_on_submit=False):

        values = {}

        # Credit Profile Section
        st.markdown(
            "<h3 class='section-header'>Credit Profile</h3>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            values["LIMIT_BAL"] = st.number_input(
                "Credit Limit (NT$)",
                min_value=0.0,
                value=float(defaults["LIMIT_BAL"]),
                step=1000.0,
                help="Credit limit amount (in New Taiwan dollars)"
            )

        with col2:
            values["SEX"] = st.selectbox(
                "Gender",
                options=[1, 2],
                index=int(defaults["SEX"]) - 1,
                help="1: Male, 2: Female",
                format_func=lambda x: "Male" if x == 1 else "Female"
            )

        with col3:
            values["EDUCATION"] = st.selectbox(
                "Education Level",
                options=[1, 2, 3, 4],
                index=int(defaults["EDUCATION"]) - 1,
                help="1: Graduate, 2: University, 3: High School, 4: Others",
                format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}.get(x, "Others")
            )

        col4, col5 = st.columns(2)

        with col4:
            values["MARRIAGE"] = st.selectbox(
                "Marital Status",
                options=[1, 2, 3],
                index=int(defaults["MARRIAGE"]) - 1,
                help="1: Married, 2: Single, 3: Others",
                format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}.get(x, "Others")
            )

        with col5:
            values["AGE"] = st.number_input(
                "Age (years)",
                min_value=20.0,
                max_value=80.0,
                value=float(defaults["AGE"]),
                step=1.0
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Repayment History Section
        st.markdown(
            "<h3 class='section-header'>Repayment History (Last 6 Months)</h3>",
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for i, field in enumerate(REPAYMENT):
            with cols[i % 3]:
                delay_label = {
                    "PAY_0": "Month 0 (Latest)",
                    "PAY_2": "Month 2",
                    "PAY_3": "Month 3",
                    "PAY_4": "Month 4",
                    "PAY_5": "Month 5",
                    "PAY_6": "Month 6"
                }.get(field, field)

                values[field] = st.selectbox(
                    f"{field} ({delay_label})",
                    options=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    index=1,  # Default to 0
                    format_func=lambda x: {
                        -1: "Paid duly",
                        0: "Payment on time",
                        1: "1 month delay",
                        2: "2 months delay",
                        3: "3 months delay",
                        4: "4 months delay",
                        5: "5 months delay",
                        6: "6 months delay",
                        7: "7+ months delay"
                    }.get(x, f"{x}+ months delay")
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Bill Amounts Section
        st.markdown(
            "<h3 class='section-header'>Bill Amounts (Last 6 Months)</h3>",
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for i, field in enumerate(BILLS):
            with cols[i % 3]:
                month_num = int(field[-1])
                values[field] = st.number_input(
                    f"{field} (Month {month_num})",
                    min_value=0.0,
                    value=float(defaults[field]),
                    step=1000.0,
                    help=f"Bill amount in month {month_num} (NT$)"
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Payment Amounts Section
        st.markdown(
            "<h3 class='section-header'>Payment Amounts (Last 6 Months)</h3>",
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for i, field in enumerate(PAYMENTS):
            with cols[i % 3]:
                month_num = int(field[-1])
                values[field] = st.number_input(
                    f"{field} (Month {month_num})",
                    min_value=0.0,
                    value=float(defaults[field]),
                    step=1000.0,
                    help=f"Payment amount in month {month_num} (NT$)"
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Derived Features Section
        st.markdown(
            "<h3 class='section-header'>Calculated Financial Features</h3>",
            unsafe_allow_html=True
        )

        # Derived features are computed from the raw input fields.
        values["TOTAL_BILL_AMT_6M"] = sum(
            values[field] for field in BILLS
        )

        values["TOTAL_PAYMENT_AMT_6M"] = sum(
            values[field] for field in PAYMENTS
        )

        values["BILL_UTILIZATION"] = (
            values["BILL_AMT1"] / values["LIMIT_BAL"]
            if values["LIMIT_BAL"] != 0
            else 0.0
        )

        values["DELAYED_MONTHS_6M"] = sum(
            1 for field in REPAYMENT if values[field] > 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total 6-Month Bills",
                f"${values['TOTAL_BILL_AMT_6M']:,.0f}"
            )

        with col2:
            st.metric(
                "Total 6-Month Payments",
                f"${values['TOTAL_PAYMENT_AMT_6M']:,.0f}"
            )

        with col3:
            st.metric(
                "Bill Utilization",
                f"{values['BILL_UTILIZATION']:.1%}"
            )

        with col4:
            st.metric(
                "Delayed Months",
                values["DELAYED_MONTHS_6M"]
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Submit Button
        submitted = st.form_submit_button(
            "🔍 Assess Customer Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            row = pd.DataFrame([values])[FEATURES]

            probability = float(
                predict_default_probabilities(row)[0]
            )

            flagged = probability >= THRESHOLD
            risk_category, risk_color = get_risk_category(probability)

            # Results Section
            st.markdown(
                "<h3 class='section-header'>Assessment Results</h3>",
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Estimated Default Probability",
                    f"{probability:.2%}",
                    delta=f"{(probability - THRESHOLD):+.2%} from threshold" if abs(probability - THRESHOLD) > 0.01 else None,
                    delta_color="inverse" if probability > THRESHOLD else "normal"
                )

            with col2:
                st.metric(
                    "Risk Category",
                    risk_category,
                    help="Based on classification threshold"
                )

            with col3:
                st.metric(
                    "Model Classification",
                    "⚠️ Flagged for Review" if flagged else "✅ Below Threshold"
                )

            # Probability Visualization
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("#### Probability Visualization")
                st.markdown("<br>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.bar(["Below Threshold", "Flagged"], [1 - probability, probability], color=["#66bb6a", "#ef5350"])
                ax.axvline(0.5, linestyle="--", color="#ffb74d", linewidth=2, label=f"Threshold: {THRESHOLD:.2f}")
                ax.set_ylabel("Probability (probability)", color="#94a3b8")
                ax.set_xlabel("Classification", color="#94a3b8")
                ax.set_title("Default Probability Visualization", color="#e6f1ff", fontweight="600")
                ax.tick_params(colors="#94a3b8", labelsize=10)
                ax.set_ylim(0, 1)
                ax.legend(facecolor='#1a2d4d', edgecolor='#233554', labelcolor='#94a3b8')
                ax.spines['bottom'].set_color('#233554')
                ax.spines['left'].set_color('#233554')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.patch.set_facecolor('#112240')
                ax.set_facecolor('#112240')

                for i, v in enumerate([1 - probability, probability]):
                    ax.text(i, v + 0.05, f"{v:.1%}", ha='center', color='#94a3b8')

                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)

            with col2:
                st.markdown("#### Risk Gauge")
                st.markdown("<br>", unsafe_allow_html=True)

                st.progress(
                    probability,
                    text=f"Risk Level: {probability:.1%}"
                )

                st.caption(
                    f"Classification threshold: **{THRESHOLD:.1%}**"
                )

            # Interpretation
            st.divider()

            if flagged:
                st.warning(
                    f"**High Risk Detected**\n\n"
                    f"The estimated default probability ({probability:.2%}) is at or above the "
                    f"classification threshold ({THRESHOLD:.1%}). This customer should be "
                    f"flagged for additional review in a real credit workflow."
                )
            else:
                st.success(
                    f"**Acceptable Risk**\n\n"
                    f"The estimated default probability ({probability:.2%}) is below the "
                    f"classification threshold ({THRESHOLD:.1%}). This customer appears to "
                    f"have acceptable credit risk based on the model."
                )

            st.info(
                "**Important Disclaimer**\n\n"
                "This educational model is not a validated lending decision system. "
                "Do not use this prediction alone to approve or reject a real credit application. "
                "Real credit decisions require comprehensive analysis, validation, fairness testing, "
                "and appropriate human review."
            )

        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")


# ============================================================
# 9. BATCH PORTFOLIO SCORING
# ============================================================

elif page == "Portfolio Scoring":
    st.subheader("Batch Portfolio Scoring")
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "**Upload Customer Portfolio**\n\n"
        "Upload a CSV containing all required model input features. "
        "The target column is optional and is never used to calculate predictions."
    )

    uploaded = st.file_uploader(
        "Choose Customer CSV File",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)

            missing = [
                col for col in FEATURES
                if col not in data.columns
            ]

            if missing:
                st.error(
                    f"❌ Missing required features: "
                    f"{', '.join(missing)}"
                )
                st.markdown(
                    "<p style='color: var(--text-secondary);'>"
                    "Required features: " + ", ".join(FEATURES) +
                    "</p>",
                    unsafe_allow_html=True
                )

            elif data.empty:
                st.warning("⚠️ The uploaded CSV is empty.")

            else:
                scored = score_customers(data)

                st.success(
                    f"Successfully scored **{len(scored):,}** records."
                )

                # Summary Metrics
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<h3 class='section-header'>Portfolio Summary</h3>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Customers",
                    f"{len(scored):,}"
                )

                high_risk_count = int(scored['Predicted_Default'].sum())
                col2.metric(
                    "High Risk Customers",
                    f"{high_risk_count:,}",
                    f"{high_risk_count / len(scored) * 100:.1f}% of portfolio"
                )

                col3.metric(
                    "Mean Predicted Probability",
                    f"{scored['Default_Probability'].mean():.2%}"
                )

                # Risk Distribution
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<h3 class='section-header'>Risk Distribution</h3>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 5))

                risk_counts = scored['Risk_Category'].value_counts()
                bars = ax.bar(
                    risk_counts.index,
                    risk_counts.values,
                    color=['#ef5350', '#66bb6a'],
                    edgecolor='none'
                )

                ax.set_ylabel('Number of Customers', color='#94a3b8')
                ax.set_xlabel('Risk Category', color='#94a3b8')
                ax.set_title('Portfolio Risk Distribution', color='#e6f1ff', fontweight='600')
                ax.tick_params(colors='#94a3b8', labelsize=10)
                ax.spines['bottom'].set_color('#233554')
                ax.spines['left'].set_color('#233554')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                for bar, count in zip(bars, risk_counts.values):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 50,
                        str(int(count)),
                        ha='center',
                        va='bottom',
                        color='#94a3b8'
                    )

                fig.patch.set_facecolor('#112240')
                ax.set_facecolor('#112240')

                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)

                # Results Table with Filtering
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<h3 class='section-header'>Customer Results</h3>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # Filter options
                filter_col1, filter_col2 = st.columns([3, 1])

                with filter_col1:
                    search_term = st.text_input(
                        "Search customer ID",
                        key="search_customer"
                    )

                with filter_col2:
                    risk_filter = st.selectbox(
                        "Filter by Risk",
                        options=["All", "High", "Below Threshold"],
                        key="risk_filter"
                    )

                # Apply filters
                display_data = scored.copy()

                if search_term:
                    try:
                        display_data = display_data[
                            display_data.index.astype(str).str.contains(search_term) |
                            display_data['LIMIT_BAL'].astype(str).str.contains(search_term)
                        ]
                    except:
                        pass

                if risk_filter != "All":
                    display_data = display_data[
                        display_data['Risk_Category'] == risk_filter
                    ]

                # Sort options
                sort_col1, sort_col2 = st.columns(2)

                with sort_col1:
                    sort_by = st.selectbox(
                        "Sort By",
                        options=["Default Probability", "Credit Limit", "Age", "Risk Category"],
                        key="sort_by"
                    )

                with sort_col2:
                    sort_order = st.selectbox(
                        "Order",
                        options=["Descending", "Ascending"],
                        key="sort_order"
                    )

                # Prepare display columns
                display_columns = [
                    "LIMIT_BAL", "SEX", "AGE", "BILL_UTILIZATION",
                    "DELAYED_MONTHS_6M", "Default_Probability", "Risk_Category"
                ]

                display_df = display_data[display_columns].copy()
                display_df["Risk_Level"] = display_df["Default_Probability"].apply(
                    lambda x: "🔴 High" if x >= THRESHOLD else "🟡 Elevated" if x >= THRESHOLD * 0.7 else "🟢 Low"
                )

                # Sort
                sort_mapping = {
                    "Default Probability": "Default_Probability",
                    "Credit Limit": "LIMIT_BAL",
                    "Age": "AGE",
                    "Risk Category": "Risk_Category"
                }

                display_df = display_df.sort_values(
                    by=sort_mapping.get(sort_by, "Default_Probability"),
                    ascending=(sort_order == "Ascending")
                )

                # Pagination
                page_size = 50
                total_pages = max(1, len(display_df) // page_size + (1 if len(display_df) % page_size > 0 else 0))

                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])

                with page_col1:
                    current_page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key="page_number"
                    )

                with page_col2:
                    st.write(f"Showing {min(page_size, len(display_df))} of {len(display_df)} records")

                with page_col3:
                    st.write(f"Page {current_page} of {total_pages}")

                start_idx = (current_page - 1) * page_size
                end_idx = min(start_idx + page_size, len(display_df))

                st.dataframe(
                    display_df.iloc[start_idx:end_idx],
                    use_container_width=True,
                    hide_index=True
                )

                # Download Button
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download All Predictions",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="credit_risk_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as exc:
            st.error(f"❌ Portfolio scoring failed: {exc}")

    else:
        st.info(
            "**Upload a CSV file**\n\n"
            "Upload a customer CSV file to generate batch predictions and risk analysis."
        )


# ============================================================
# 10. MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":
    st.subheader("Model Performance and Explainability")
    st.markdown("<br>", unsafe_allow_html=True)

    # Model Performance Metrics
    st.markdown(
        "<h3 class='section-header'>Model Performance Metrics</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test ROC-AUC",
        f"{float(metadata.get('roc_auc', 0)):.4f}",
        help="Area under the Receiver Operating Characteristic curve (higher is better)"
    )

    col2.metric(
        "Classification Threshold",
        f"{THRESHOLD:.2f}",
        help="Probability threshold for high-risk classification"
    )

    col3.metric(
        "Model",
        metadata.get("model_name", "XGBoost"),
        help="XGBoost Classifier with gradient boosting"
    )

    # Feature Importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 class='section-header'>Top 15 Feature Importance</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    try:
        feature_names = preprocessor.get_feature_names_out()
        importance_values = classifier.feature_importances_

        if len(feature_names) != len(importance_values):
            st.warning(
                "⚠️ The feature-name count does not match the model's feature-importance count. "
                "Check that the saved preprocessor and classifier were exported from the same pipeline.",
                icon="⚠️"
            )

        else:
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance_values
            }).sort_values(
                "Importance",
                ascending=False
            )

            # Create readable feature labels
            readable_labels = {
                "LIMIT_BAL": "Credit Limit",
                "SEX": "Gender",
                "EDUCATION": "Education Level",
                "MARRIAGE": "Marital Status",
                "AGE": "Age",
                "PAY_0": "Latest Payment Delay",
                "PAY_2": "2-Month Payment Delay",
                "PAY_3": "3-Month Payment Delay",
                "PAY_4": "4-Month Payment Delay",
                "PAY_5": "5-Month Payment Delay",
                "PAY_6": "6-Month Payment Delay",
                "BILL_AMT1": "Most Recent Bill",
                "BILL_AMT2": "2-Month Old Bill",
                "BILL_AMT3": "3-Month Old Bill",
                "BILL_AMT4": "4-Month Old Bill",
                "BILL_AMT5": "5-Month Old Bill",
                "BILL_AMT6": "6-Month Old Bill",
                "PAY_AMT1": "Most Recent Payment",
                "PAY_AMT2": "2-Month Old Payment",
                "PAY_AMT3": "3-Month Old Payment",
                "PAY_AMT4": "4-Month Old Payment",
                "PAY_AMT5": "5-Month Old Payment",
                "PAY_AMT6": "6-Month Old Payment",
                "TOTAL_BILL_AMT_6M": "Total 6-Month Bills",
                "TOTAL_PAYMENT_AMT_6M": "Total 6-Month Payments",
                "BILL_UTILIZATION": "Bill Utilization Ratio",
                "DELAYED_MONTHS_6M": "Delayed Months (6M)"
            }

            # Create DataFrame with readable labels
            importance_display = pd.DataFrame({
                "Feature": [readable_labels.get(f, f) for f in feature_names],
                "Importance": importance_values
            }).sort_values("Importance", ascending=False).head(15)

            st.dataframe(
                importance_display,
                use_container_width=True,
                hide_index=True
            )

            # Feature Importance Chart
            st.markdown("<br>", unsafe_allow_html=True)

            top_features = importance_display.head(15).copy()
            ordered = top_features.sort_values("Importance")

            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.barh(
                ordered["Feature"],
                ordered["Importance"],
                color="#64b5f6",
                edgecolor="none"
            )

            ax.set_xlabel("XGBoost Feature Importance (importance score)", color="#94a3b8")
            ax.set_title("Top 15 Most Important Features", color="#e6f1ff", fontweight="600")
            ax.tick_params(colors="#94a3b8", labelsize=10)
            ax.invert_yaxis()

            for bar, value in zip(bars, ordered["Importance"]):
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height()/2,
                    f"{value:.4f}",
                    ha='left',
                    va='center',
                    color="#94a3b8",
                    fontsize=9
                )

            ax.spines['bottom'].set_color('#233554')
            ax.spines['left'].set_color('#233554')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.patch.set_facecolor('#112240')
            ax.set_facecolor('#112240')

            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)

    except Exception as exc:
        st.error(f"❌ Unable to calculate feature importance: {exc}")

    # Interpretation Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 class='section-header'>Interpretation Guide</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "**Key Predictors of Credit Risk**\n\n"
        "Based on the model analysis, the following factors show the strongest "
        "correlation with default probability:\n\n"
        "1. **Repayment History** (PAY_0, PAY_2, etc.) - Most significant predictor\n"
        "2. **Bill Utilization** (BILL_UTILIZATION) - Credit usage relative to limit\n"
        "3. **Payment Amounts** (PAY_AMT1, PAY_AMT2) - Recent payment behavior\n"
        "4. **Credit Limit** (LIMIT_BAL) - Overall credit exposure\n"
        "5. **Delay Patterns** (DELAYED_MONTHS_6M) - Consistent payment issues\n\n"
        "These features help the model identify customers at higher risk of default.",
        icon="🔍"
    )

    st.warning(
        "**Important Distinction**\n\n"
        "**Feature Importance ≠ Causation**\n\n"
        "The displayed feature importance describes how the model uses input features "
        "to make predictions; it does not establish causal relationships between these "
        "factors and default behavior. Correlation does not imply causation.\n\n"
        "The ROC-AUC score (0.778) indicates good discriminatory power, but this is a "
        "portfolio project for educational and analytical use only. Real credit decisions "
        "require comprehensive validation, fairness testing, governance, and appropriate "
        "human review."
    )

    st.caption(
        "This is an educational machine-learning prototype. Predictions should not be used "
        "alone for real credit decisions."
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: var(--text-secondary);">
        <p style="margin: 0; font-size: 0.9rem;">
            Credit Risk Analytics Platform | Educational Prototype
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;">
            XGBoost Classifier | Scikit-learn Pipeline | Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
