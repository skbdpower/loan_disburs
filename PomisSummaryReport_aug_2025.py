import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import os

# Enhanced Custom CSS for modern KPI boxes and UI
st.markdown("""
<style>
    .main .block-container { padding: 2.5rem; }
    h1 { font-size: 28px !important; color: #1a3c34; }
    h2 { font-size: 22px !important; color: #2e5b52; }
    h3 { font-size: 18px !important; color: #2e5b52; }
    .kpi-box { 
        background: linear-gradient(135deg, #f6f9fc 0%, #e8f4f8 100%); 
        border-radius: 12px; 
        padding: 25px; 
        box-shadow: 0 6px 12px rgba(0,0,0,0.1); 
        margin-bottom: 20px; 
        border: 1px solid #d1e7dd;
        transition: transform 0.2s ease-in-out;
    }
    .kpi-box:hover { transform: translateY(-2px); }
    .kpi-box .stMetric label { 
        font-size: 18px !important; 
        font-weight: bold; 
        color: #1a3c34; 
        text-align: center;
    }
    .kpi-box .stMetric span { 
        font-size: 26px !important; 
        color: #0a6e5a; 
        text-align: center;
        font-weight: 600;
    }
    .stDataFrame { font-size: 14px !important; }
    .divider { margin: 25px 0; border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_file = "PomisSummaryReport_aug_2025.csv"
    if not os.path.exists(csv_file):
        st.error(f"Error: The file '{csv_file}' was not found in the working directory. Please ensure it is present.")
        return None, None
    
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        
        # Compute KPIs
        df['Net_Member_Growth'] = df['New_Member_Admission_August_2025'] - df['Member_Cancellation_August_2025']
        df['Member_Growth_Rate'] = (df['Net_Member_Growth'] / df['New_Member_Admission_August_2025'].replace(0, np.nan)) * 100
        df['Total_Recovered'] = df['Recovered_Regular_August_2025'] + df['Recovered_Due_August_2025']
        df['Recovery_Rate'] = (df['Total_Recovered'] / df['Reg_Loan_Recoverable_August_2025'].replace(0, np.nan)) * 100
        df['Overdue_Outstanding'] = df['As_on_August_2025_Overdue_Loan_Outstanding']
        df['PAR_Rate'] = (df['Overdue_Outstanding'] / df['As on August_2025 Cummulative Loan Disbursement'].replace(0, np.nan)) * 100
        df['Collection_Rate'] = (df['As_on_August_2025_Cummulative_Loan_Collection'] / df['As on August_2025 Cummulative Loan Disbursement'].replace(0, np.nan)) * 100
        
        kpis = [
            'Net_Member_Growth',
            'Member_Growth_Rate',
            'Borrower_August_2025',
            'Fully_Paid_Borrower_August_2025',
            'Recovery_Rate',
            'PAR_Rate',
            'Collection_Rate',
            'As on August_2025 Cummulative Loan Disbursement'
        ]
        
        return df, kpis
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

def format_kpi_value(kpi, value):
    if np.isnan(value):
        return "N/A"
    if kpi == 'As on August_2025 Cummulative Loan Disbursement':
        return f"{value:,.0f} BDT"
    elif kpi in ['Member_Growth_Rate', 'Recovery_Rate', 'PAR_Rate', 'Collection_Rate']:
        return f"{value:.2f}%"
    else:
        return f"{value:,.0f}"

kpi_tooltips = {
    'Net_Member_Growth': 'New admissions minus cancellations (membership growth)',
    'Member_Growth_Rate': 'Percentage growth in members (Net Growth / New Admissions)',
    'Borrower_August_2025': 'Total active borrowers in August 2025',
    'Fully_Paid_Borrower_August_2025': 'Borrowers who fully repaid loans in August 2025',
    'Recovery_Rate': 'Percentage of recoverable loans collected',
    'PAR_Rate': 'Portfolio at Risk: Overdue loans as a percentage of total disbursements',
    'Collection_Rate': 'Percentage of cumulative loans collected',
    'As on August_2025 Cummulative Loan Disbursement': 'Total loans disbursed as of August 2025'
}

def tune_best_model(X, y):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ''
    results = {}
    
    for name, model in models.items():
        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
            score = grid_search.best_score_
        else:
            score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
            model.fit(X, y)
        
        results[name] = {'model': model, 'r2': score}
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    
    # Compute MSE for best model
    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return best_model, best_name, best_score, mse, results

st.title("Pomis Summary Analysis Dashboard with Predictions - August 2025")
st.markdown("Domain-wise KPIs, Model-Tuned Predictions, and Modern Visualizations")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Load data
df, kpis = load_data()
if df is None or kpis is None:
    st.stop()

# Enhanced KPIs Section with New Box Style
st.header("Key Performance Indicators (KPIs)")
domain_summary = df.groupby('Domain')[kpis].agg({
    'Net_Member_Growth': 'sum',
    'Member_Growth_Rate': 'mean',
    'Borrower_August_2025': 'sum',
    'Fully_Paid_Borrower_August_2025': 'sum',
    'Recovery_Rate': 'mean',
    'PAR_Rate': 'mean',
    'Collection_Rate': 'mean',
    'As on August_2025 Cummulative Loan Disbursement': 'sum'
}).round(2)

kpi_cols = st.columns(4)
for i, kpi in enumerate(kpis):
    total = domain_summary[kpi].sum() if kpi not in ['Member_Growth_Rate', 'Recovery_Rate', 'PAR_Rate', 'Collection_Rate'] else domain_summary[kpi].mean()
    formatted_value = format_kpi_value(kpi, total)
    with kpi_cols[i % 4]:
        st.markdown(f'<div class="kpi-box">', unsafe_allow_html=True)
        st.metric(label=kpi.replace('_', ' ').title(), value=formatted_value, help=kpi_tooltips.get(kpi, ''))
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Domain-wise Visualizations
st.header("Domain-wise KPI Visualizations")
for kpi in kpis:
    st.subheader(f"{kpi.replace('_', ' ').title()}")
    fig = px.bar(
        domain_summary.reset_index(),
        x='Domain',
        y=kpi,
        title=f"{kpi.replace('_', ' ').title()} by Domain",
        text=kpi,
        color='Domain',
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='seaborn'
    )
    if kpi == 'As on August_2025 Cummulative Loan Disbursement':
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='auto', textfont_size=14)
    else:
        fig.update_traces(texttemplate='%{text:.2f}', textposition='auto', textfont_size=14)
    fig.update_layout(
        title_font_size=22,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14,
        showlegend=False,
        xaxis_title="Domain",
        yaxis_title=kpi.replace('_', ' ')
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Analysis & Predictions Section
with st.expander("Analysis & Predictions for September 2025", expanded=True):
    st.header("Analysis & Predictions")
    
    # Prepare data for modeling
    features = ['Branch_Code', 'Borrower_August_2025', 'Fully_Paid_Borrower_August_2025', 'Recovery_Rate', 'PAR_Rate', 'Collection_Rate']
    X = df[features].fillna(0)  # Handle NaNs
    y = df['Net_Member_Growth']
    
    # Tune and select best model
    best_model, best_name, best_r2, mse, results = tune_best_model(X, y)
    y_pred = best_model.predict(X)
    
    # Display model performance
    st.info(f"Best Model: {best_name}\nR² Score: {best_r2:.3f}\nMean Squared Error: {mse:.2f}")
    
    # Predict September 2025
    future_branch_code = df['Branch_Code'].max() + 1
    X_future = X.copy()
    X_future['Branch_Code'] = future_branch_code
    sep_pred = best_model.predict(X_future)
    df['Predicted_Net_Growth_Sep_2025'] = sep_pred * 1.025  # 2.5% seasonal uplift
    overall_pred = df['Predicted_Net_Growth_Sep_2025'].sum()
    
    st.write(f"**Predicted Total Net Member Growth (Sep 2025)**: {overall_pred:.0f}")
    
    # Domain-wise predictions
    domain_pred = df.groupby('Domain')['Predicted_Net_Growth_Sep_2025'].sum().round(0)
    st.write("**Domain-wise Predicted Net Member Growth (Sep 2025):**")
    st.dataframe(domain_pred, use_container_width=True)
    
    # Prediction Plot 1: Scatter
    st.subheader("Net Member Growth: Actual vs Predicted")
    fig_scatter = px.scatter(
        x=df['Branch_Code'], y=df['Net_Member_Growth'],
        labels={'x': 'Branch Code (Time Proxy)', 'y': 'Net Member Growth'},
        title="Net Member Growth: August vs Predicted September",
        template='seaborn'
    )
    fig_scatter.add_scatter(
        x=df['Branch_Code'], y=y_pred, mode='lines', name='Fitted Trend', line=dict(color='red')
    )
    fig_scatter.add_scatter(
        x=[future_branch_code], y=[overall_pred / len(df)], mode='markers', name='Sep Prediction',
        marker=dict(size=15, color='green')
    )
    fig_scatter.update_layout(
        title_font_size=22, xaxis_title_font_size=18, yaxis_title_font_size=18,
        xaxis_tickfont_size=14, yaxis_tickfont_size=14
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Prediction Plot 2: Correlation Heatmap
    st.subheader("KPI Correlations")
    corr = df[['Net_Member_Growth', 'Borrower_August_2025', 'Recovery_Rate', 'Collection_Rate']].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmin=-1, zmax=1, text=corr.values.round(2),
        texttemplate='%{text}', textfont=dict(size=14)
    ))
    fig_heatmap.update_layout(
        title="Correlation of Key KPIs", title_font_size=22,
        xaxis_title_font_size=18, yaxis_title_font_size=18,
        xaxis_tickfont_size=14, yaxis_tickfont_size=14,
        template='seaborn'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Prediction Plot 3: Domain Bar Chart
    st.subheader("Domain-wise Growth: Actual vs Predicted")
    domain_actual = df.groupby('Domain')['Net_Member_Growth'].sum()
    domain_df = pd.DataFrame({'Actual': domain_actual, 'Predicted': domain_pred})
    fig_bar = px.bar(
        domain_df.reset_index(), x='Domain', y=['Actual', 'Predicted'],
        barmode='group', title="Actual vs Predicted Net Member Growth by Domain",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='seaborn'
    )
    fig_bar.update_traces(texttemplate='%{value:.0f}', textposition='auto', textfont_size=14)
    fig_bar.update_layout(
        title_font_size=22, xaxis_title_font_size=18, yaxis_title_font_size=18,
        xaxis_tickfont_size=14, yaxis_tickfont_size=14,
        yaxis_title="Net Member Growth"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Top Performing Branches
st.header("Top Performing Branches (by Net Member Growth)")
top_branches = df.nlargest(10, 'Net_Member_Growth')[['Branch_Name', 'Domain', 'Zone', 'Net_Member_Growth', 'Recovery_Rate', 'Collection_Rate']].round(2)
st.dataframe(top_branches, use_container_width=True)

# Scatter Plot: Recovery vs Collection Rate
st.subheader("Recovery Rate vs Collection Rate for Top Branches")
fig_scatter = px.scatter(
    top_branches, x='Recovery_Rate', y='Collection_Rate', size='Net_Member_Growth',
    hover_name='Branch_Name', title="Recovery Rate vs Collection Rate (Top Branches)",
    text='Branch_Name', template='seaborn'
)
fig_scatter.update_traces(textposition='top center', textfont_size=12)
fig_scatter.update_layout(
    title_font_size=22, xaxis_title_font_size=18, yaxis_title_font_size=18,
    xaxis_tickfont_size=14, yaxis_tickfont_size=14
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Raw Data Table
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
#st.markdown("**Note:** Ensure 'PomisSummaryReport_aug_2025.csv' is in the same directory. Run with `streamlit run dashboard_with_predictions.py`.")
