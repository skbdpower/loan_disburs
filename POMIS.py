import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set the page configuration for a wide layout
st.set_page_config(layout="wide", page_title="Interactive POMIS July_2025 Dashboard")

# --- Custom CSS for a refreshed, clean look ---
st.markdown("""
<style>
@import url('https://fonts.com/css2?family=Roboto:wght@300;400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}
.main-header {
    font-size: 3.5rem;
    font-weight: 700;
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 20px;
    letter-spacing: -1px;
}
.subheader {
    font-size: 2rem;
    font-weight: 500;
    color: #333333;
    border-bottom: 3px solid #f0f0f0;
    padding-bottom: 10px;
    margin-top: 40px;
}
.kpi-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
    margin: 10px;
    transition: transform 0.3s, box-shadow 0.3s;
}
.kpi-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
}
.kpi-title {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 10px;
    text-transform: uppercase;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.kpi-value {
    font-size: 2.8rem;
    font-weight: 700;
    color: #000;
}
.green { color: #2ecc71; }
.red { color: #e74c3c; }
.blue { color: #3498db; }
.purple { color: #9b59b6; }
.orange { color: #e67e22; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>POMIS JULY_2025 Interactive Dashboard</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2rem; color:#888;'>Overall Performance Metrics for July 2025</p>", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """
    Loads data from a CSV file and cleans the column names.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df

# Load the data from the user-provided CSV file.
try:
    df = load_data('Pomis_Summary_Report_july_2025.csv')
except FileNotFoundError:
    st.error("Error: The file 'Pomis_Summary_Report_july_2025.xlsx - Pomis_Summary_Report.csv' was not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# --- KPI Calculations ---
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

# Calculate the Recovered Rate
recovered_rate = 0
if total_loan_recoverable > 0:
    recovered_rate = (total_recovered_regular / total_loan_recoverable) * 100

# --- Helper function to render a styled KPI card ---
def display_kpi_card(title, value, value_color):
    """
    Renders a single KPI card with a dynamic title and value color.
    """
    if isinstance(value, (int, float)):
        if "rate" in title.lower():
            formatted_value = f"{value:,.2f}%"
        elif "amount" in title.lower() or "loan" in title.lower() or "outstanding" in title.lower() or "due" in title.lower() or "collection" in title.lower() or "disbursement" in title.lower():
            formatted_value = f"{value:,.0f}"
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

# --- Overall Key Performance Indicators with Structured and Resized Layout ---
st.markdown("<h2 class='subheader'>Overall Key Performance Indicators</h2>", unsafe_allow_html=True)

# Row 1: The most critical metrics side-by-side
col1, col2 = st.columns([2, 1])
with col1:
    display_kpi_card("Total Outstanding", total_outstanding, "blue")
with col2:
    display_kpi_card("Recovered Rate %", recovered_rate, "purple")

# Row 2: Financial health metrics
col3, col4, col5 = st.columns(3)
with col3:
    display_kpi_card("Total Due", total_total_due, "red")
with col4:
    display_kpi_card("Overdue Outstanding", total_overdue_outstanding, "red")
with col5:
    display_kpi_card("Bad Loan (365+)", total_bad_loan, "red")

# Row 3: Membership and loan activity
col6, col7, col8 = st.columns(3)
with col6:
    display_kpi_card("New Member Admission", total_new_members, "blue")
with col7:
    display_kpi_card("Disbursement Borrower", total_disbursement_borrower, "green")
with col8:
    display_kpi_card("Fully Paid Borrower", total_fully_paid_borrower, "green")

# Row 4: Newly added metrics
col9, col10, col11, col12 = st.columns(4)
with col9:
    display_kpi_card("Member Cancellation", total_member_cancellations, "red")
with col10:
    display_kpi_card("July Due Amount", total_due_july, "red")
with col11:
    display_kpi_card("New Due July", total_new_due_july, "red")
with col12:
    display_kpi_card("Total Due Loanee", total_total_due_loanee, "red")

st.markdown("<h2 class='subheader'>Overall Performance & Financial Flow</h2>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)

# Chart 1: Waterfall Chart for Financial Flow
with col_a:
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Financial Flow", orientation = "v",
        measure = ["relative", "relative"],
        x = ["Recovered", "Due"],
        textposition = "outside",
        text = [f"{total_recovered_regular:,.0f}", f"{total_total_due:,.0f}"],
        y = [total_recovered_regular, total_total_due],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#e74c3c"}},
        increasing = {"marker":{"color":"#2ecc71"}},
        totals = {"marker":{"color":"#3498db"}},
    ))
    fig_waterfall.add_trace(go.Waterfall(
        name="Loan Recoverable",
        orientation="v",
        measure=["total"],
        x=["Loan Recoverable"],
        textposition="outside",
        text=[f"{total_loan_recoverable:,.0f}"],
        y=[total_loan_recoverable],
        connector={"line":{"color":"rgb(63, 63, 63)"}}
    ))
    fig_waterfall.update_layout(
        title_text='Loan Recovery Flow',
        showlegend = False,
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

# Chart 2: Gauge Chart for Recovered Rate
with col_b:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = recovered_rate,
        title = {'text': "Loan Recovered Rate %"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#3498db"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 85], 'color': "#e67e22"},
                {'range': [85, 100], 'color': "#2ecc71"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    fig_gauge.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("<h2 class='subheader'>Key Metric Relationships & Comparisons</h2>", unsafe_allow_html=True)
col_c, col_d = st.columns(2)

# Chart 3: Pie Chart for Cumulative Disbursement vs. Collection
with col_c:
    cumulative_data = pd.DataFrame({
        'Metric': ['Cumulative Disbursement', 'Cumulative Collection'],
        'Amount': [total_cummulative_disbursement, total_cummulative_collection]
    })
    fig_pie_cumulative = px.pie(
        cumulative_data,
        values='Amount',
        names='Metric',
        title='Overall Cumulative Disbursement vs. Collection',
        color_discrete_sequence=['#3498db', '#2ecc71']
    )
    fig_pie_cumulative.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie_cumulative, use_container_width=True)

# Chart 4: Pie Chart for New Members vs. Cancellations
with col_d:
    member_data = pd.DataFrame({
        'Status': ['New Members', 'Cancellations'],
        'Count': [total_new_members, total_member_cancellations]
    })
    fig_member_pie = px.pie(
        member_data,
        values='Count',
        names='Status',
        title='New Member Admissions vs. Cancellations',
        color='Status',
        color_discrete_map={
            'New Members': '#3498db',
            'Cancellations': '#e74c3c'
        }
    )
    fig_member_pie.update_traces(textposition='inside', textinfo='percent+label+value')
    st.plotly_chart(fig_member_pie, use_container_width=True)

st.markdown("<h2 class='subheader'>Loan Health Overview</h2>", unsafe_allow_html=True)
col_e, col_f = st.columns(2)

# Chart 5: Pie Chart for Overdue vs. Regular Outstanding
with col_e:
    outstanding_data = pd.DataFrame({
        'Status': ['Regular Outstanding', 'Overdue Outstanding'],
        'Amount': [total_outstanding - total_overdue_outstanding, total_overdue_outstanding]
    })
    fig_pie = px.pie(
        outstanding_data,
        values='Amount',
        names='Status',
        title='Proportion of Overdue Outstanding',
        color='Status',
        color_discrete_map={
            'Regular Outstanding': '#3498db',
            'Overdue Outstanding': '#e74c3c'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# Chart 6: Pie Chart for Disbursement vs. Fully Paid Borrowers
with col_f:
    borrower_data = pd.DataFrame({
        'Status': ['Disbursement Borrowers', 'Fully Paid Borrowers'],
        'Count': [total_disbursement_borrower, total_fully_paid_borrower]
    })
    fig_borrower_pie = px.pie(
        borrower_data,
        values='Count',
        names='Status',
        title='Disbursement vs. Fully Paid Borrowers',
        color='Status',
        color_discrete_map={
            'Disbursement Borrowers': '#2ecc71',
            'Fully Paid Borrowers': '#e67e22'
        }
    )
    fig_borrower_pie.update_traces(textposition='inside', textinfo='percent+label+value')
    st.plotly_chart(fig_borrower_pie, use_container_width=True)

# Use an expander to hide the raw data by default
st.divider()
with st.expander("View Raw Data"):
    st.subheader("Full Data Table")
    st.dataframe(df)
