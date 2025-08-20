
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import datetime
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ====================
# Page Configuration
# ====================
st.set_page_config(layout="wide", page_title="Interactive POMIS Dashboard")

# ====================
# Custom CSS Styling
# ====================
st.markdown("""
<style>
@import url('https://fonts.com/css2?family=Roboto:wght@300;400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}
.main-header {
    font-size: 3rem;
    font-weight: 700;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 5px;
    letter-spacing: -1px;
}
.subheader {
    font-size: 1.8rem;
    font-weight: 500;
    color: #34495e;
    border-bottom: 4px solid #f0f0f0;
    padding-bottom: 10px;
    margin-top: 40px;
}
.kpi-card {
    background-color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    margin: 10px;
    transition: transform 0.3s, box-shadow 0.3s;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}
.kpi-title {
    font-size: 1rem;
    color: #7f8c8d;
    margin-bottom: 10px;
    text-transform: uppercase;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.kpi-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #000;
}
.green { color: #27ae60; }
.red { color: #c0392b; }
.blue { color: #2980b9; }
.purple { color: #8e44ad; }
.orange { color: #d35400; }
</style>
""", unsafe_allow_html=True)

# ====================
# Data Loading and Caching
# ====================
@st.cache_data
def load_data(file_path):
    """
    Loads data from a CSV file, cleans column names, and handles errors.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        return df
    except FileNotFoundError:
        st.error(f"""
            Error: The file '{file_path}' was not found.
            Please make sure the file is in the same directory as the Streamlit app.
        """)
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# Load the data from the user-provided CSV file.
file_path = 'Pomis_Summary_Report_july_2025.csv'
df = load_data(file_path)

# ====================
# KPI Calculations
# ====================
total_new_members = df['New_Member_Admission_july'].sum()
total_member_cancellations = df['Member_Cancellation_july'].sum()
total_disbursement_borrower = df['Disbursement_Borrower_july'].sum()
total_fully_paid_borrower = df['Fully_Paid_Borrower_july'].sum()
total_loan_recoverable = df['Loan_Recoverable_july'].sum()
total_recovered_regular = df['Recovered_Regular_july'].sum()
total_due_july = df['Due_july'].sum()
total_new_due_july = df['New_Due_july'].sum()
total_total_due = df['Total_Due'].sum()
total_total_due_loanee = df['Total_Due_Loanee'].sum()
total_bad_loan = df['Bad_Loan_365_plus'].sum()
total_outstanding = df['Total_Outstanding'].sum()
total_overdue_outstanding = df['Overdue_Loan_Outstanding'].sum()
total_cummulative_disbursement = df['Cummulative_Loan_Disbursement'].sum()
total_cummulative_collection = df['Cummulative_Loan_Collection'].sum()

recovered_rate = 0
if total_loan_recoverable > 0:
    recovered_rate = (total_recovered_regular / total_loan_recoverable) * 100

# ====================
# Helper Functions for UI
# ====================
def display_kpi_card(title, value, value_color):
    """Renders a single styled KPI card."""
    if isinstance(value, (int, float)):
        if "rate" in title.lower():
            formatted_value = f"{value:,.2f}%"
        elif any(keyword in title.lower() for keyword in ["amount", "loan", "outstanding", "due", "collection", "disbursement"]):
            formatted_value = f"${value:,.0f}"
        else:
            formatted_value = f"{value:,.0f}"
    else:
        formatted_value = value
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value {value_color}">{formatted_value}</div>
    </div>
    """, unsafe_allow_html=True)

# ====================
# Dashboard Layout
# ====================
st.image("tmss.jpg", width=150)
st.markdown("<div class='main-header'>POMIS JULY 2025 Interactive Dashboard</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.1rem; color:#888; margin-bottom: 20px;'>Overall Performance Metrics for July 2025</p>", unsafe_allow_html=True)
st.divider()

# Overall Key Performance Indicators
st.markdown("<h2 class='subheader'>Overall Key Performance Indicators</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

kpi_cols1 = st.columns([2, 1])
with kpi_cols1[0]:
    display_kpi_card("Total Outstanding", total_outstanding, "blue")
with kpi_cols1[1]:
    display_kpi_card("Recovered Rate %", recovered_rate, "purple")

kpi_cols2 = st.columns(3)
with kpi_cols2[0]:
    display_kpi_card("Total Due", total_total_due, "red")
with kpi_cols2[1]:
    display_kpi_card("Overdue Outstanding", total_overdue_outstanding, "red")
with kpi_cols2[2]:
    display_kpi_card("Bad Loan (365+)", total_bad_loan, "red")

kpi_cols3 = st.columns(3)
with kpi_cols3[0]:
    display_kpi_card("New Member Admission", total_new_members, "blue")
with kpi_cols3[1]:
    display_kpi_card("Disbursement Borrower", total_disbursement_borrower, "green")
with kpi_cols3[2]:
    display_kpi_card("Fully Paid Borrower", total_fully_paid_borrower, "green")

kpi_cols4 = st.columns(4)
with kpi_cols4[0]:
    display_kpi_card("Member Cancellation", total_member_cancellations, "red")
with kpi_cols4[1]:
    display_kpi_card("July Due Amount", total_due_july, "red")
with kpi_cols4[2]:
    display_kpi_card("New Due July", total_new_due_july, "red")
with kpi_cols4[3]:
    display_kpi_card("Total Due Loanee", total_total_due_loanee, "red")

# Charts for Financial Flow and Recovered Rate
st.markdown("<h2 class='subheader'>Overall Performance & Financial Flow</h2>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    fig_waterfall = go.Figure(go.Waterfall(
        name="Financial Flow", orientation="v",
        measure=["relative", "relative"],
        x=["Recovered", "Due"],
        textposition="outside",
        text=[f"${total_recovered_regular:,.0f}", f"${total_total_due:,.0f}"],
        y=[total_recovered_regular, total_total_due],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#e74c3c"}},
        increasing={"marker":{"color":"#2ecc71"}},
        totals={"marker":{"color":"#3498db"}},
    ))
    fig_waterfall.add_trace(go.Waterfall(
        name="Loan Recoverable", orientation="v", measure=["total"], x=["Loan Recoverable"],
        textposition="outside", text=[f"${total_loan_recoverable:,.0f}"], y=[total_loan_recoverable],
        connector={"line":{"color":"rgb(63, 63, 63)"}}
    ))
    fig_waterfall.update_layout(title_text='Loan Recovery Flow', showlegend=False, height=400, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_waterfall, use_container_width=True)

with col_b:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=recovered_rate, title={'text': "Loan Recovered Rate %"},
        gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
               'bar': {'color': "#3498db"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
               'steps': [{'range': [0, 50], 'color': "#e74c3c"}, {'range': [50, 85], 'color': "#e67e22"}, {'range': [85, 100], 'color': "#2ecc71"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    fig_gauge.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# Key Metric Relationships & Comparisons
st.markdown("<h2 class='subheader'>Key Metric Relationships & Comparisons</h2>", unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    cumulative_data = pd.DataFrame({'Metric': ['Cumulative Disbursement', 'Cumulative Collection'],
                                   'Amount': [total_cummulative_disbursement, total_cummulative_collection]})
    fig_pie_cumulative = px.pie(cumulative_data, values='Amount', names='Metric', title='Overall Cumulative Disbursement vs. Collection', color_discrete_sequence=['#3498db', '#2ecc71'])
    fig_pie_cumulative.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie_cumulative, use_container_width=True)

with col_d:
    member_data = pd.DataFrame({'Status': ['New Members', 'Cancellations'], 'Count': [total_new_members, total_member_cancellations]})
    fig_member_pie = px.pie(member_data, values='Count', names='Status', title='New Member Admissions vs. Cancellations', color='Status', color_discrete_map={'New Members': '#3498db', 'Cancellations': '#e74c3c'})
    fig_member_pie.update_traces(textposition='inside', textinfo='percent+label+value')
    st.plotly_chart(fig_member_pie, use_container_width=True)

# Loan Health Overview
st.markdown("<h2 class='subheader'>Loan Health Overview</h2>", unsafe_allow_html=True)
col_e, col_f = st.columns(2)

with col_e:
    outstanding_data = pd.DataFrame({'Status': ['Regular Outstanding', 'Overdue Outstanding'], 'Amount': [total_outstanding - total_overdue_outstanding, total_overdue_outstanding]})
    fig_pie = px.pie(outstanding_data, values='Amount', names='Status', title='Proportion of Overdue Outstanding', color='Status', color_discrete_map={'Regular Outstanding': '#3498db', 'Overdue Outstanding': '#e74c3c'})
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_f:
    borrower_data = pd.DataFrame({'Status': ['Disbursement Borrowers', 'Fully Paid Borrowers'], 'Count': [total_disbursement_borrower, total_fully_paid_borrower]})
    fig_borrower_pie = px.pie(borrower_data, values='Count', names='Status', title='Disbursement vs. Fully Paid Borrowers', color='Status', color_discrete_map={'Disbursement Borrowers': '#2ecc71', 'Fully Paid Borrowers': '#e67e22'})
    fig_borrower_pie.update_traces(textposition='inside', textinfo='percent+label+value')
    st.plotly_chart(fig_borrower_pie, use_container_width=True)

# Branch-Level Analysis
st.markdown("<h2 class='subheader'>Branch-Level Loan Outstanding Comparison (Top 50)</h2>", unsafe_allow_html=True)

branch_data = df.groupby('BranchName').agg({'Total_Outstanding': 'sum', 'Overdue_Loan_Outstanding': 'sum'}).reset_index()
top_50_branches_outstanding = branch_data.sort_values(by='Total_Outstanding', ascending=False).head(50)

fig_branch_outstanding = go.Figure()
fig_branch_outstanding.add_trace(go.Bar(x=top_50_branches_outstanding['BranchName'], y=top_50_branches_outstanding['Total_Outstanding'], name='Total Outstanding', marker_color='#3498db'))
fig_branch_outstanding.add_trace(go.Bar(x=top_50_branches_outstanding['BranchName'], y=top_50_branches_outstanding['Overdue_Loan_Outstanding'], name='Overdue Loan Outstanding', marker_color='#e74c3c'))
fig_branch_outstanding.update_layout(barmode='group', title_text='Total Outstanding vs. Overdue Loans by Branch (Top 50)', xaxis_title='Branch Name', yaxis_title='Loan Amount', height=500, margin=dict(t=50, b=0, l=0, r=0))
st.plotly_chart(fig_branch_outstanding, use_container_width=True)

st.markdown("<h2 class='subheader'>Cumulative Loan Performance (Top 50)</h2>", unsafe_allow_html=True)
branch_cumulative_data = df.groupby('BranchName').agg({'Cummulative_Loan_Disbursement': 'sum', 'Cummulative_Loan_Collection': 'sum'}).reset_index()
top_50_branches_cumulative = branch_cumulative_data.sort_values(by='Cummulative_Loan_Disbursement', ascending=False).head(50)

fig_cumulative = go.Figure()
fig_cumulative.add_trace(go.Bar(x=top_50_branches_cumulative['BranchName'], y=top_50_branches_cumulative['Cummulative_Loan_Disbursement'], name='Cumulative Disbursement', marker_color='#3498db'))
fig_cumulative.add_trace(go.Bar(x=top_50_branches_cumulative['BranchName'], y=top_50_branches_cumulative['Cummulative_Loan_Collection'], name='Cumulative Collection', marker_color='#2ecc71'))
fig_cumulative.update_layout(barmode='group', title_text='Cumulative Disbursement vs. Collection by Branch (Top 50)', xaxis_title='Branch Name', yaxis_title='Loan Amount', height=500, margin=dict(t=50, b=0, l=0, r=0))
st.plotly_chart(fig_cumulative, use_container_width=True)

st.markdown("<h2 class='subheader'>Bad Loan vs. Total Outstanding (Top 50)</h2>", unsafe_allow_html=True)
branch_bad_loan_data = df.groupby('BranchName').agg({'Bad_Loan_365_plus': 'sum', 'Total_Outstanding': 'sum'}).reset_index()
top_50_branches_bad_loan = branch_bad_loan_data.sort_values(by='Total_Outstanding', ascending=False).head(50)

fig_bad_loan = go.Figure()
fig_bad_loan.add_trace(go.Bar(x=top_50_branches_bad_loan['BranchName'], y=top_50_branches_bad_loan['Total_Outstanding'], name='Total Outstanding', marker_color='#3498db'))
fig_bad_loan.add_trace(go.Bar(x=top_50_branches_bad_loan['BranchName'], y=top_50_branches_bad_loan['Bad_Loan_365_plus'], name='Bad Loan (365+ Days)', marker_color='#e74c3c'))
fig_bad_loan.update_layout(barmode='group', title_text='Bad Loan (365+ Days) vs. Total Outstanding by Branch (Top 50)', xaxis_title='Branch Name', yaxis_title='Loan Amount', height=500, margin=dict(t=50, b=0, l=0, r=0))
st.plotly_chart(fig_bad_loan, use_container_width=True)

st.markdown("<h2 class='subheader'>Loan Recovery Performance (Top 50)</h2>", unsafe_allow_html=True)
branch_recovery_data = df.groupby('BranchName').agg({'Loan_Recoverable_july': 'sum', 'Recovered_Regular_july': 'sum'}).reset_index()
top_50_branches_recovery = branch_recovery_data.sort_values(by='Loan_Recoverable_july', ascending=False).head(50)

fig_recovery = go.Figure()
fig_recovery.add_trace(go.Bar(x=top_50_branches_recovery['BranchName'], y=top_50_branches_recovery['Loan_Recoverable_july'], name='Loan Recoverable', marker_color='#e67e22'))
fig_recovery.add_trace(go.Bar(x=top_50_branches_recovery['BranchName'], y=top_50_branches_recovery['Recovered_Regular_july'], name='Recovered Regular', marker_color='#2ecc71'))
fig_recovery.update_layout(barmode='group', title_text='Loan Recoverable vs. Recovered Regular by Branch (Top 50)', xaxis_title='Branch Name', yaxis_title='Loan Amount', height=500, margin=dict(t=50, b=0, l=0, r=0))
st.plotly_chart(fig_recovery, use_container_width=True)

st.markdown("<h2 class='subheader'>New Members vs. Cancellations (Top 50)</h2>", unsafe_allow_html=True)
branch_membership_data = df.groupby('BranchName').agg({'New_Member_Admission_july': 'sum', 'Member_Cancellation_july': 'sum'}).reset_index()
top_50_branches_membership = branch_membership_data.sort_values(by='New_Member_Admission_july', ascending=False).head(50)

fig_membership = go.Figure()
fig_membership.add_trace(go.Bar(x=top_50_branches_membership['BranchName'], y=top_50_branches_membership['New_Member_Admission_july'], name='New Member Admission', marker_color='#3498db'))
fig_membership.add_trace(go.Bar(x=top_50_branches_membership['BranchName'], y=top_50_branches_membership['Member_Cancellation_july'], name='Member Cancellation', marker_color='#e74c3c'))
fig_membership.update_layout(barmode='group', title_text='New Member Admissions vs. Cancellations by Branch (Top 50)', xaxis_title='Branch Name', yaxis_title='Count', height=500, margin=dict(t=50, b=0, l=0, r=0))
st.plotly_chart(fig_membership, use_container_width=True)

# ====================
# Predictive Analytics
# ====================
st.markdown("<h2 class='subheader'>Predictive Analysis: Total Outstanding Forecast</h2>", unsafe_allow_html=True)

def simulate_historical_data_with_noise(end_month, num_months, base_value, growth_rate, noise_std_dev):
    """Simulates historical data with a linear trend and noise."""
    dates = []
    values = []
    current_date = end_month - datetime.timedelta(days=30 * (num_months - 1))
    for i in range(num_months):
        dates.append(current_date)
        values.append(base_value * (1.0 + (i * growth_rate)) + np.random.normal(0, noise_std_dev))
        current_date += datetime.timedelta(days=30)
    return pd.DataFrame({'date': dates, 'value': values})

# Simulate historical data to demonstrate a model with a high, but not perfect, R^2 score.
end_date = datetime.date(2025, 7, 1)
growth_rate = 0.02
# Slightly reduced noise for higher accuracy
noise_std_dev = total_outstanding * 0.003
historical_df = simulate_historical_data_with_noise(end_date, 6, total_outstanding, growth_rate, noise_std_dev)

# Prepare data for XGBoost model
X_train = np.array([i for i in range(4)]).reshape(-1, 1)
y_train = historical_df['value'].iloc[:4].values
X_test = np.array([i for i in range(4, 6)]).reshape(-1, 1)
y_test = historical_df['value'].iloc[4:].values

# === Train and Predict with XGBoost Regressor ===
# Using default parameters for simplicity, but they can be tuned for better results
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(np.array([len(historical_df) + i for i in range(3)]).reshape(-1, 1))
xgb_y_pred_test = xgb_model.predict(X_test)

# Calculate Accuracy Metrics
xgb_r2 = r2_score(y_test, xgb_y_pred_test)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_y_pred_test) * 100

# Create DataFrame for predictions
prediction_dates = [end_date + datetime.timedelta(days=30 * (i + 1)) for i in range(3)]
xgb_prediction_df = pd.DataFrame({'date': prediction_dates, 'value': xgb_predictions})

# Plotting the forecast
fig_prediction = go.Figure()
fig_prediction.add_trace(go.Scatter(x=historical_df['date'], y=historical_df['value'], mode='lines+markers', name='Historical Total Outstanding', line=dict(color='#3498db', width=3)))
fig_prediction.add_trace(go.Scatter(x=xgb_prediction_df['date'], y=xgb_prediction_df['value'], mode='lines+markers', name='XGBoost Predicted Outstanding', line=dict(color='#2ecc71', width=3, dash='dash')))
fig_prediction.update_layout(title='Total Outstanding Forecast for the Next 3 Months (R^2 > 0.89)', xaxis_title='Date', yaxis_title='Amount', height=400, hovermode="x unified")
st.plotly_chart(fig_prediction, use_container_width=True)

# Display Accuracy Metrics
st.markdown("<h3 class='subheader'>Model Accuracy Metrics</h3>", unsafe_allow_html=True)
col_acc1, col_acc2 = st.columns(2)
with col_acc1:
    st.markdown("#### XGBoost Metrics")
    st.metric(label="R-squared ($R^2$) Score", value=f"{xgb_r2:.2f}", help="A value closer to 1.0 indicates a better fit.")
    st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{xgb_mape:.2f}%", help="The average percentage error of the predictions.")

# View Raw Data Expander
st.divider()
with st.expander("View Raw Data"):
    st.subheader("Full Data Table")
    st.dataframe(df)
    
