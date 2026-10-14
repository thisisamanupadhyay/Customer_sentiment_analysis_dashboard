"""
app.py
------
Customer Sentiment Analysis Dashboard
Built with Streamlit, Pandas, NLTK VADER, and Plotly.
Designed for college internship and beginner-friendly data science projects.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentiment_analysis import process_reviews_dataframe, analyze_sentiment

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean cards and modern look
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        text-align: left;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .badge-pos { color: #16a34a; font-weight: 600; }
    .badge-neg { color: #dc2626; font-weight: 600; }
    .badge-neu { color: #d97706; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Loads the pre-packaged customer reviews CSV file."""
    default_path = os.path.join(os.path.dirname(__file__), "data", "customer_reviews.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        return df
    else:
        st.error(f"Default dataset not found at {default_path}")
        return pd.DataFrame()


def render_kpi_card(title: str, value: str, subtext: str = ""):
    """Renders a modern metric card with HTML styling."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Sidebar: Upload & Filters
# ---------------------------------------------------------
st.sidebar.title("🛠️ Dashboard Controls")
st.sidebar.markdown("Upload a CSV file or use the built-in customer dataset.")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Reviews CSV",
    type=["csv"],
    help="CSV must contain at least a 'Review' column. Optional columns: Customer_ID, Rating, Date, Product_Category",
)

# Load data based on user upload or default
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        raw_df = load_default_data()
else:
    raw_df = load_default_data()
    st.sidebar.info("Using default dataset: `customer_reviews.csv`")

if raw_df.empty:
    st.warning("No data available to display. Please upload a valid CSV file.")
    st.stop()

# Perform Sentiment Analysis
with st.spinner("Analyzing sentiments..."):
    try:
        df = process_reviews_dataframe(raw_df)
    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
        st.stop()

# Ensure standard column fallbacks for custom CSVs
if "Rating" not in df.columns:
    df["Rating"] = np.nan
if "Product_Category" not in df.columns:
    df["Product_Category"] = "General"
if "Customer_ID" not in df.columns:
    df["Customer_ID"] = [f"CUST-{1001 + i}" for i in range(len(df))]
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

# Filter 1: Product Category
categories = sorted([cat for cat in df["Product_Category"].dropna().unique()])
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=categories,
    default=categories,
    help="Filter reviews by product category",
)

# Filter 2: Sentiment
sentiments = ["Positive", "Neutral", "Negative"]
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=sentiments,
    default=sentiments,
    help="Filter by sentiment classification",
)

# Filter 3: Rating (if ratings exist)
if not df["Rating"].dropna().empty:
    valid_ratings = sorted([int(r) for r in df["Rating"].dropna().unique() if r in [1, 2, 3, 4, 5]])
    if valid_ratings:
        selected_ratings = st.sidebar.multiselect(
            "Star Rating",
            options=valid_ratings,
            default=valid_ratings,
            help="Filter by customer rating (1 to 5 stars)",
        )
    else:
        selected_ratings = None
else:
    selected_ratings = None

# Filter 4: Date Range (if valid dates exist)
if "Date" in df.columns and not df["Date"].dropna().empty:
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None

# Apply Filters
filtered_df = df.copy()

if selected_categories:
    filtered_df = filtered_df[filtered_df["Product_Category"].isin(selected_categories)]

if selected_sentiments:
    filtered_df = filtered_df[filtered_df["Sentiment"].isin(selected_sentiments)]

if selected_ratings:
    filtered_df = filtered_df[filtered_df["Rating"].isin(selected_ratings)]

if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[
        (filtered_df["Date"] >= start_dt) & (filtered_df["Date"] <= end_dt)
    ]

# ---------------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------------
st.title("📊 Customer Sentiment Analysis Dashboard")
st.markdown(
    "Gain actionable insights from customer feedback with automated sentiment scoring, distribution metrics, and category trends."
)
st.write("")

# ---------------------------------------------------------
# KPI Overview Metrics
# ---------------------------------------------------------
total_reviews = len(filtered_df)
pos_count = (filtered_df["Sentiment"] == "Positive").sum()
neg_count = (filtered_df["Sentiment"] == "Negative").sum()
neu_count = (filtered_df["Sentiment"] == "Neutral").sum()

pos_pct = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
neg_pct = (neg_count / total_reviews * 100) if total_reviews > 0 else 0
neu_pct = (neu_count / total_reviews * 100) if total_reviews > 0 else 0

avg_rating = (
    filtered_df["Rating"].dropna().mean()
    if not filtered_df["Rating"].dropna().empty
    else None
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    render_kpi_card("Total Reviews", f"{total_reviews:,}", f"Active filters applied")

with kpi_col2:
    render_kpi_card(
        "Positive Reviews",
        f"{pos_count:,}",
        f"<span class='badge-pos'>{pos_pct:.1f}%</span> of total",
    )

with kpi_col3:
    render_kpi_card(
        "Neutral Reviews",
        f"{neu_count:,}",
        f"<span class='badge-neu'>{neu_pct:.1f}%</span> of total",
    )

with kpi_col4:
    render_kpi_card(
        "Negative Reviews",
        f"{neg_count:,}",
        f"<span class='badge-neg'>{neg_pct:.1f}%</span> of total",
    )

with kpi_col5:
    rating_display = f"{avg_rating:.2f} / 5.0" if avg_rating is not None else "N/A"
    render_kpi_card("Average Rating", rating_display, "⭐ Customer satisfaction")

st.write("")
st.markdown("---")

if total_reviews == 0:
    st.warning("⚠️ No reviews match the selected filter criteria. Please broaden your sidebar filters.")
    st.stop()

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
st.subheader("📈 Sentiment Insights & Trends")

sentiment_color_map = {
    "Positive": "#16a34a",  # emerald green
    "Neutral": "#d97706",   # amber yellow
    "Negative": "#dc2626",  # red
}

chart_row1_col1, chart_row1_col2 = st.columns(2)

# Chart 1: Donut Chart - Sentiment Distribution
with chart_row1_col1:
    sentiment_counts = (
        filtered_df["Sentiment"].value_counts().reset_index()
    )
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig_donut = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.55,
        title="<b>Sentiment Distribution</b>",
        color="Sentiment",
        color_discrete_map=sentiment_color_map,
    )
    fig_donut.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#ffffff", width=2)),
    )
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, b=30, l=20, r=20),
        height=360,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Chart 2: Bar Chart - Reviews by Rating
with chart_row1_col2:
    if not filtered_df["Rating"].dropna().empty:
        rating_counts = (
            filtered_df["Rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        rating_counts.columns = ["Rating", "Count"]
        rating_counts["Rating_Label"] = rating_counts["Rating"].astype(str) + " Stars"

        fig_rating = px.bar(
            rating_counts,
            x="Rating_Label",
            y="Count",
            title="<b>Review Count by Star Rating</b>",
            text="Count",
            color="Rating_Label",
            color_discrete_sequence=["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"],
        )
        fig_rating.update_traces(textposition="outside")
        fig_rating.update_layout(
            xaxis_title="Rating",
            yaxis_title="Number of Reviews",
            showlegend=False,
            margin=dict(t=50, b=30, l=20, r=20),
            height=360,
        )
        st.plotly_chart(fig_rating, use_container_width=True)
    else:
        st.info("No rating data available in the dataset.")

chart_row2_col1, chart_row2_col2 = st.columns(2)

# Chart 3: Sentiment by Product Category
with chart_row2_col1:
    cat_sent = (
        filtered_df.groupby(["Product_Category", "Sentiment"])
        .size()
        .reset_index(name="Count")
    )
    fig_cat = px.bar(
        cat_sent,
        x="Product_Category",
        y="Count",
        color="Sentiment",
        title="<b>Sentiment by Product Category</b>",
        barmode="group",
        color_discrete_map=sentiment_color_map,
    )
    fig_cat.update_layout(
        xaxis_title="Category",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(t=50, b=40, l=20, r=20),
        height=380,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# Chart 4: Sentiment Trend Over Date
with chart_row2_col2:
    if "Date" in filtered_df.columns and not filtered_df["Date"].dropna().empty:
        trend_df = (
            filtered_df.dropna(subset=["Date"])
            .set_index("Date")
            .groupby([pd.Grouper(freq="W"), "Sentiment"])
            .size()
            .reset_index(name="Count")
        )
        fig_trend = px.line(
            trend_df,
            x="Date",
            y="Count",
            color="Sentiment",
            title="<b>Weekly Sentiment Trend Over Time</b>",
            markers=True,
            color_discrete_map=sentiment_color_map,
        )
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Review Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=50, b=40, l=20, r=20),
            height=380,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No date information available for trend visualization.")

st.markdown("---")

# ---------------------------------------------------------
# Customer Reviews Table with Keyword Search
# ---------------------------------------------------------
st.subheader("📋 Customer Reviews Data Table")

col_search, col_dl = st.columns([3, 1])

with col_search:
    search_keyword = st.text_input(
        "🔎 Search reviews by keyword:",
        placeholder="e.g. battery, fabric, return, sound, comfortable...",
    )

# Filter table by keyword search if provided
table_df = filtered_df.copy()
if search_keyword.strip():
    table_df = table_df[
        table_df["Review"].str.contains(search_keyword, case=False, na=False)
    ]
    st.caption(f"Found {len(table_df)} review(s) matching '{search_keyword}'")

with col_dl:
    st.write("")
    st.write("")
    csv_export = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv_export,
        file_name="sentiment_analyzed_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Format Date column for clean presentation
display_df = table_df.copy()
if "Date" in display_df.columns:
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

# Select display columns
desired_cols = ["Customer_ID", "Review", "Rating", "Product_Category", "Sentiment", "Sentiment_Score"]
existing_cols = [c for c in desired_cols if c in display_df.columns]
if "Date" in display_df.columns and "Date" not in existing_cols:
    existing_cols.insert(3, "Date")

st.dataframe(
    display_df[existing_cols],
    use_container_width=True,
    height=350,
)

st.markdown("---")

# ---------------------------------------------------------
# Interactive Live Review Tester (Great for viva / demos)
# ---------------------------------------------------------
st.subheader("🧪 Live Review Sentiment Tester")
st.caption("Try entering a custom review to test the VADER sentiment classifier in real time.")

test_col1, test_col2 = st.columns([3, 1])

with test_col1:
    user_test_input = st.text_input(
        "Enter sample review text:",
        value="The battery life is amazing and charging is super fast!",
        placeholder="Type any product feedback here...",
    )

with test_col2:
    st.write("")
    st.write("")
    test_btn = st.button("Analyze Sentiment", use_container_width=True)

if user_test_input:
    label, score = analyze_sentiment(user_test_input)
    
    if label == "Positive":
        box_color = "success"
        emoji = "😃"
    elif label == "Negative":
        box_color = "error"
        emoji = "😡"
    else:
        box_color = "warning"
        emoji = "😐"

    st.info(
        f"**Prediction:** {emoji} **{label}** | **Compound Score:** `{score}` (Thresholds: Positive ≥ 0.05, Negative ≤ -0.05, Neutral in between)"
    )
