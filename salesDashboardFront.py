import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# ── Page config (mobile-friendly) ─────────────────────────────────────────
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Generate dataset ───────────────────────────────────────────────────────
@st.cache_data
def generate_data(seed=42):
    np.random.seed(seed)
    N = 6000

    categories  = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys']
    products = {
        'Electronics':   ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smart Watch'],
        'Clothing':      ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Sneakers'],
        'Home & Garden': ['Sofa', 'Lamp', 'Vacuum Cleaner', 'Plant Pot', 'Curtains'],
        'Sports':        ['Running Shoes', 'Yoga Mat', 'Dumbbells', 'Bicycle', 'Tennis Racket'],
        'Books':         ['Fiction Novel', 'Textbook', 'Biography', 'Cookbook', 'Self-Help'],
        'Toys':          ['LEGO Set', 'Board Game', 'Action Figure', 'Puzzle', 'Remote Car']
    }
    regions   = ['North', 'South', 'East', 'West', 'Central']
    payments  = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
    segments  = ['Premium', 'Regular', 'New']
    base_prices = {
        'Electronics': 500, 'Clothing': 60, 'Home & Garden': 150,
        'Sports': 120, 'Books': 25, 'Toys': 40
    }

    dates      = pd.date_range('2022-01-01', '2024-12-31', periods=N)
    cat_col    = np.random.choice(categories, N)
    prod_col   = [np.random.choice(products[c]) for c in cat_col]
    unit_price = np.array([base_prices[c] * np.random.uniform(0.5, 2.0) for c in cat_col])
    quantity   = np.random.randint(1, 20, N)
    discount   = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], N)
    revenue    = (unit_price * quantity * (1 - discount)).round(2)
    profit     = (revenue * np.random.uniform(0.3, 0.6, N)).round(2)

    df = pd.DataFrame({
        'order_id':       range(1, N+1),
        'order_date':     dates,
        'category':       cat_col,
        'product':        prod_col,
        'region':         np.random.choice(regions, N),
        'segment':        np.random.choice(segments, N, p=[0.2, 0.5, 0.3]),
        'payment_method': np.random.choice(payments, N),
        'quantity':       quantity,
        'unit_price':     unit_price.round(2),
        'discount':       discount,
        'revenue':        revenue,
        'profit':         profit
    })

    df['year']          = df['order_date'].dt.year
    df['month']         = df['order_date'].dt.month
    df['month_name']    = df['order_date'].dt.strftime('%b')
    df['profit_margin'] = (df['profit'] / df['revenue'] * 100).round(2)

    # Clean
    df['discount'] = df['discount'].fillna(df['discount'].median())
    df['segment']  = df['segment'].fillna('Regular')
    df = df.drop_duplicates()
    Q1, Q3 = df['revenue'].quantile(0.25), df['revenue'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['revenue'] >= Q1 - 3*IQR) & (df['revenue'] <= Q3 + 3*IQR)]

    return df

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Dashboard")
st.sidebar.markdown("---")

# Real-time: refresh button regenerates data with new seed
if st.sidebar.button("🔄 Refresh Data (Real-Time)"):
    st.cache_data.clear()

df = generate_data()

# Filters
st.sidebar.markdown("### Filters")

all_categories = ['All'] + sorted(df['category'].unique().tolist())
all_regions    = ['All'] + sorted(df['region'].unique().tolist())
all_years      = ['All'] + sorted(df['year'].unique().tolist())
all_segments   = ['All'] + sorted(df['segment'].unique().tolist())

sel_category = st.sidebar.selectbox("Category",   all_categories)
sel_region   = st.sidebar.selectbox("Region",     all_regions)
sel_year     = st.sidebar.selectbox("Year",       all_years)
sel_segment  = st.sidebar.selectbox("Segment",    all_segments)

# Search
search_term = st.sidebar.text_input("🔍 Search Product", "")

# Date range
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()
date_from = st.sidebar.date_input("Date From", value=min_date, min_value=min_date, max_value=max_date)
date_to   = st.sidebar.date_input("Date To",   value=max_date, min_value=min_date, max_value=max_date)

# Apply filters
filtered = df.copy()
if sel_category != 'All':
    filtered = filtered[filtered['category'] == sel_category]
if sel_region != 'All':
    filtered = filtered[filtered['region'] == sel_region]
if sel_year != 'All':
    filtered = filtered[filtered['year'] == sel_year]
if sel_segment != 'All':
    filtered = filtered[filtered['segment'] == sel_segment]
if search_term:
    filtered = filtered[filtered['product'].str.contains(search_term, case=False, na=False)]


# ── Main layout ────────────────────────────────────────────────────────────
st.title("📊 Sales Analytics Dashboard")
st.markdown(f"Showing **{len(filtered):,}** records after filters")
st.markdown("---")

# ── KPIs ───────────────────────────────────────────────────────────────────
if len(filtered) == 0:
    st.warning("No data for the selected filters.")
    st.stop()

total_revenue  = filtered['revenue'].sum()
total_profit   = filtered['profit'].sum()
avg_revenue    = filtered['revenue'].mean()
customer_count = filtered['order_id'].nunique()
best_product   = filtered.groupby('product')['revenue'].sum().idxmax()
avg_margin     = filtered['profit_margin'].mean()

monthly_rev = filtered.groupby(['year','month'])['revenue'].sum().reset_index()
monthly_rev = monthly_rev.sort_values(['year','month'])
if len(monthly_rev) >= 2:
    mom_growth = (monthly_rev['revenue'].iloc[-1] - monthly_rev['revenue'].iloc[-2]) / monthly_rev['revenue'].iloc[-2] * 100
else:
    mom_growth = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenue",    f"${total_revenue:,.0f}")
c2.metric("📈 Total Profit",     f"${total_profit:,.0f}")
c3.metric("🛒 Avg Order Value",  f"${avg_revenue:,.0f}")
c4.metric("👥 Customer Count",   f"{customer_count:,}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("📊 Profit Margin",    f"{avg_margin:.1f}%")
c6.metric("📅 Monthly Growth",   f"{mom_growth:+.1f}%")
c7.metric("🏆 Best Product",     best_product)
c8.metric("🏷️ Best Category",   filtered.groupby('category')['revenue'].sum().idxmax())

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────
st.subheader("📈 Revenue & Profit Trend")
monthly = filtered.groupby(['year','month'])[['revenue','profit']].sum().reset_index()
monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))
monthly = monthly.sort_values('date')
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=monthly['date'], y=monthly['revenue'],
                               mode='lines+markers', name='Revenue', line=dict(color='#2196F3')))
fig_line.add_trace(go.Scatter(x=monthly['date'], y=monthly['profit'],
                               mode='lines+markers', name='Profit',  line=dict(color='#4CAF50', dash='dash')))
fig_line.update_layout(template='plotly_white', height=350)
st.plotly_chart(fig_line, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Revenue by Category")
    cat_rev = filtered.groupby('category')['revenue'].sum().sort_values(ascending=False).reset_index()
    fig_bar = px.bar(cat_rev, x='category', y='revenue',
                     color='revenue', color_continuous_scale='Blues',
                     template='plotly_white')
    fig_bar.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("🌍 Revenue by Region")
    reg_rev = filtered.groupby('region')['revenue'].sum().reset_index()
    fig_reg = px.bar(reg_rev, x='region', y='revenue',
                     color='revenue', color_continuous_scale='Oranges',
                     template='plotly_white')
    fig_reg.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_reg, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🧩 Revenue by Segment (Donut)")
    seg_rev = filtered.groupby('segment')['revenue'].sum().reset_index()
    fig_pie = px.pie(seg_rev, names='segment', values='revenue',
                     hole=0.5, template='plotly_white',
                     color_discrete_sequence=['#2196F3','#4CAF50','#FF9800'])
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    st.subheader("🔵 Revenue vs Profit (Scatter)")
    fig_scat = px.scatter(filtered, x='revenue', y='profit', color='category',
                          opacity=0.5, template='plotly_white',
                          trendline='ols')
    fig_scat.update_layout(height=350)
    st.plotly_chart(fig_scat, use_container_width=True)

st.subheader("🌡️ Revenue Heatmap (Category × Month)")
heat = filtered.groupby(['category','month'])['revenue'].sum().unstack(fill_value=0)
heat.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig_heat = px.imshow(heat/1000, text_auto='.0f', color_continuous_scale='YlOrRd',
                     labels=dict(color='Revenue K$'), template='plotly_white')
fig_heat.update_layout(height=320)
st.plotly_chart(fig_heat, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.subheader("📊 Profit Margin by Category (Box)")
    fig_box = px.box(filtered, x='category', y='profit_margin',
                     color='category', template='plotly_white')
    fig_box.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with col6:
    st.subheader("📊 Correlation Heatmap")
    corr = filtered[['quantity','unit_price','discount','revenue','profit','profit_margin']].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1, template='plotly_white')
    fig_corr.update_layout(height=350)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ── Business Insights ──────────────────────────────────────────────────────
st.subheader("💡 Business Insights")
i1, i2 = st.columns(2)
with i1:
    st.markdown("**Top 3 categories by revenue:**")
    top3 = filtered.groupby('category')['revenue'].sum().sort_values(ascending=False).head(3)
    for cat, val in top3.items():
        st.markdown(f"- {cat}: **${val:,.0f}**")

    st.markdown("**Best months:**")
    top_months = filtered.groupby('month_name')['revenue'].sum().sort_values(ascending=False).head(3)
    for m, val in top_months.items():
        st.markdown(f"- {m}: **${val:,.0f}**")

with i2:
    st.markdown("**Best regions:**")
    top_reg = filtered.groupby('region')['revenue'].sum().sort_values(ascending=False)
    for r, val in top_reg.items():
        st.markdown(f"- {r}: **${val:,.0f}**")

    st.markdown("**Underperforming categories (low margin):**")
    low = filtered.groupby('category')['profit_margin'].mean().sort_values().head(2)
    for cat, val in low.items():
        st.markdown(f"- {cat}: **{val:.1f}%**")

st.markdown("---")

# ── ML Section ─────────────────────────────────────────────────────────────
st.subheader("🤖 Machine Learning — Revenue Prediction (Bonus)")

with st.expander("Show ML Results", expanded=False):
    le = LabelEncoder()
    ml = filtered.copy()
    for col in ['category','region','segment','payment_method']:
        ml[col+'_enc'] = le.fit_transform(ml[col].astype(str))

    features = ['quantity','unit_price','discount','month','year',
                'category_enc','region_enc','segment_enc','payment_method_enc']
    X = ml[features]
    y = ml['revenue']

    if len(X) < 50:
        st.warning("Not enough data for ML with current filters.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        lr_r2  = r2_score(y_test, lr.predict(X_test))
        rf_r2  = r2_score(y_test, rf.predict(X_test))
        lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
        rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

        m1, m2 = st.columns(2)
        m1.metric("Linear Regression R²", f"{lr_r2:.4f}", f"MAE: ${lr_mae:,.0f}")
        m2.metric("Random Forest R²",     f"{rf_r2:.4f}", f"MAE: ${rf_mae:,.0f}")

        # Actual vs Predicted
        y_pred = rf.predict(X_test)
        fig_ml = px.scatter(x=y_test[:300], y=y_pred[:300],
                            labels={'x':'Actual','y':'Predicted'},
                            title=f'Actual vs Predicted (RF, R²={rf_r2:.3f})',
                            template='plotly_white', opacity=0.5)
        fig_ml.add_shape(type='line',
                         x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max(),
                         line=dict(color='red', dash='dash'))
        st.plotly_chart(fig_ml, use_container_width=True)

        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        fig_imp = px.bar(importance, x='importance', y='feature', orientation='h',
                         title='Feature Importance', template='plotly_white',
                         color='importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ── AI Chatbot ─────────────────────────────────────────────────────────────
st.subheader("🤖 AI Chatbot — Ask About Your Data (Bonus)")
st.caption("Powered by Groq API (free). Get your key at: console.groq.com")

groq_key = st.text_input("Enter your Groq API Key", type="password",
                          placeholder="gsk_...")

if groq_key:
    # Build data summary for context
    data_summary = f"""
    Sales dataset summary:
    - Total records: {len(filtered):,}
    - Total revenue: ${filtered['revenue'].sum():,.0f}
    - Total profit: ${filtered['profit'].sum():,.0f}
    - Date range: {filtered['order_date'].min().date()} to {filtered['order_date'].max().date()}
    - Categories: {', '.join(filtered['category'].unique())}
    - Regions: {', '.join(filtered['region'].unique())}
    - Best category: {filtered.groupby('category')['revenue'].sum().idxmax()}
    - Best product: {filtered.groupby('product')['revenue'].sum().idxmax()}
    - Avg profit margin: {filtered['profit_margin'].mean():.1f}%
    - Top region: {filtered.groupby('region')['revenue'].sum().idxmax()}
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about the sales data...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    headers = {
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": f"You are a data analyst assistant. Here is the current dataset summary:\n{data_summary}\nAnswer questions about this sales data concisely."},
                            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                        ],
                        "max_tokens": 500
                    }
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=15
                    )
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}. Check your Groq API key.")
else:
    st.info("Enter your Groq API key above to enable the AI chatbot.")

st.markdown("---")
st.caption("Sales Analytics Dashboard | Data Analysis and Visualization Course")
