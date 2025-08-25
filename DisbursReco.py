import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from prophet import Prophet
import plotly.graph_objects as go

# Set page configuration for a wide, clean layout
st.set_page_config(
    page_title="Loan Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a beautiful, modern look with slightly larger text and improved color scheme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f5;
        color: #2c3e50;
    }
    .main-header {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2.0rem; }
    h3 { font-size: 1.5rem; }
    .stMetric > div {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .stMetric > div:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stMetric > div > div:first-child {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 5px;
    }
    .stMetric > div > div:nth-child(2) {
        font-size: 28px;
        font-weight: bold;
        color: #34495e;
    }
    .stMetric > div > div:nth-child(3) {
        font-size: 14px;
        color: #adb5bd;
    }
    .chart-container {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .st-emotion-cache-1c7y2c1 {
        padding-top: 1rem;
    }
    .st-emotion-cache-1c7y2c1 .st-emotion-cache-1v0603s {
        gap: 1.0rem;
    }
    .st-emotion-cache-1c7y2c1 .st-emotion-cache-1v0603s > div {
        flex: 1 1 0%;
        max-width: 100%;
        min-width: 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_name):
    """
    Loads the loan data from a CSV file.
    """
    try:
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_name}' was not found. Please ensure it's in the correct directory.")
        return None

# The name of the uploaded file
FILE_NAME = "Dhaka_0043_Loan_Disbursment_Recovery_Dec_2023_July_2025.csv"
data = load_data(FILE_NAME)

if data is not None:
    # --- Data Preprocessing ---
    try:
        data['Disbursement_Date'] = pd.to_datetime(data['Disbursement_Date'])
    except Exception as e:
        st.error(f"Error converting 'Disbursement_Date' to datetime: {e}")
        st.stop()

    # --- Sidebar for Interactive Filters ---
    st.sidebar.header("Dashboard Filters")
    
    # Filter by Loan Product
    loan_products = data['Loan_Product'].unique()
    selected_products = st.sidebar.multiselect(
        "Select Loan Product(s):",
        options=loan_products,
        default=loan_products
    )

    # Filter by Disbursement Date Range
    min_date = data['Disbursement_Date'].min().date()
    max_date = data['Disbursement_Date'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data[
            (data['Loan_Product'].isin(selected_products)) &
            (data['Disbursement_Date'].dt.date >= start_date) &
            (data['Disbursement_Date'].dt.date <= end_date)
        ]
    else:
        st.warning("Please select a valid date range.")
        filtered_data = pd.DataFrame()

    # --- Dashboard Title and Summary ---
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    
    # Add logo and title
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        # NOTE: You MUST replace this placeholder URL with the public URL of your tmss.jpg image.
        # Streamlit cannot access local files on your computer directly.
        st.image(
            "tmss.jpg", 
            width=150
        )
    with col_title:
        st.title("Dhaka Branch Disbursment-Recovery Dec2023-July2025 Performance Dashboard")
        st.markdown("Use the sidebar filters to dynamically analyze the loan data.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not filtered_data.empty:
        # --- KPI Calculations and Display ---
        st.header("Key Performance SUM")
        
        total_disbursement = filtered_data['Disbursement_Amount'].sum()
        total_repay_amount = filtered_data['Repay_Amount'].sum()
        total_recovered = filtered_data['Recovered_Amount'].sum()
        total_service_charge = filtered_data['Service_Charge'].sum()
        
        recovery_rate = (total_recovered / total_repay_amount) * 100 if total_repay_amount > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(label="Total Disbursement", value=f"{total_disbursement:,.0f}")
        with col2:
            st.metric(label="Total Recovered", value=f"{total_recovered:,.0f}")
        with col3:
            st.metric(label="Total Repayable", value=f"{total_repay_amount:,.0f}")
        with col4:
            st.metric(label="Recovery Rate", value=f"{recovery_rate:.2f}%")
        with col5:
            st.metric(label="Total Service Charge", value=f"{total_service_charge:,.0f}")

        st.markdown("---")

        # --- Visualizations Section ---
        st.header("Visualizations")
        
        # New Plot 1: Recovery Rate over Time
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Recovery Rate Over Time")
        recovery_over_time = filtered_data.groupby(filtered_data['Disbursement_Date'].dt.to_period('M')).agg(
            Total_Recovered=('Recovered_Amount', 'sum'),
            Total_Repayable=('Repay_Amount', 'sum')
        ).reset_index()
        recovery_over_time['Recovery_Rate'] = (recovery_over_time['Total_Recovered'] / recovery_over_time['Total_Repayable']) * 100
        recovery_over_time['Disbursement_Date'] = recovery_over_time['Disbursement_Date'].astype(str)
        
        # Using a distinct, single color for the line chart
        fig_recovery_rate_line = px.line(
            recovery_over_time,
            x='Disbursement_Date',
            y='Recovery_Rate',
            title='Recovery Rate Over Time',
            labels={'Disbursement_Date': 'Month', 'Recovery_Rate': 'Recovery Rate (%)'},
            markers=True
        )
        fig_recovery_rate_line.update_traces(line=dict(color='#1f77b4')) # A professional blue
        fig_recovery_rate_line.update_layout(hovermode="x unified")
        fig_recovery_rate_line.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_recovery_rate_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Updated Plot 2: Disbursement vs Recovery by Loan Product with better colors
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Disbursement vs. Recovered Amount by Loan Product")
        disb_rec_by_product = filtered_data.groupby('Loan_Product').agg(
            Total_Disbursement=('Disbursement_Amount', 'sum'),
            Total_Recovered=('Recovered_Amount', 'sum')
        ).reset_index()
        
        fig_disb_rec_bar = go.Figure()
        fig_disb_rec_bar.add_trace(go.Bar(
            x=disb_rec_by_product['Loan_Product'],
            y=disb_rec_by_product['Total_Disbursement'],
            name='Total Disbursement',
            marker_color='#5A82B4', # A professional, deep blue
            text=disb_rec_by_product['Total_Disbursement'],
            textposition='outside',
            texttemplate='%{y:,.0f}'
        ))
        fig_disb_rec_bar.add_trace(go.Bar(
            x=disb_rec_by_product['Loan_Product'],
            y=disb_rec_by_product['Total_Recovered'],
            name='Total Recovered',
            marker_color='#A3C1E0', # A lighter shade of blue for contrast
            text=disb_rec_by_product['Total_Recovered'],
            textposition='outside',
            texttemplate='%{y:,.0f}'
        ))
        fig_disb_rec_bar.update_layout(
            barmode='group',
            title='Disbursement vs. Recovery by Loan Product',
            xaxis_title='Loan Product',
            yaxis_title='Amount',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_disb_rec_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot 3: Disbursement Distribution by Loan Cycle
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Disbursement Amount Distribution by Loan Cycle")
        # Using a qualitative palette for the box plot
        fig_disb_cycle_box = px.box(
            filtered_data,
            x='Loan_Cycle',
            y='Disbursement_Amount',
            title='Disbursement Amount Distribution by Loan Cycle',
            labels={'Loan_Cycle': 'Loan Cycle', 'Disbursement_Amount': 'Disbursement Amount'},
            color='Loan_Cycle',
            color_discrete_sequence=px.colors.qualitative.D3
        )
        st.plotly_chart(fig_disb_cycle_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot 4: Disbursement by Loan Product (Bar) with improved colors
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Disbursement by Loan Product")
        disbursement_by_product = filtered_data.groupby('Loan_Product')['Disbursement_Amount'].sum().reset_index()
        fig_disbursement_bar = px.bar(
            disbursement_by_product,
            x='Loan_Product',
            y='Disbursement_Amount',
            title='Total Disbursement Amount by Loan Product',
            labels={'Loan_Product': 'Loan Product', 'Disbursement_Amount': 'Disbursement Amount'},
            text='Disbursement_Amount',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_disbursement_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont_size=14)
        st.plotly_chart(fig_disbursement_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Plot 5: Recovery by Loan Product (Pie) with improved colors
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Recovery by Loan Product")
        recovered_by_product = filtered_data.groupby('Loan_Product')['Recovered_Amount'].sum().reset_index()
        fig_recovered = px.pie(
            recovered_by_product,
            values='Recovered_Amount',
            names='Loan_Product',
            title='Breakdown of Total Recovered Amount',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_recovered.update_traces(textinfo='percent+value', texttemplate='%{percent}<br>%{value:,.0f}', textfont_size=16)
        st.plotly_chart(fig_recovered, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot 6: Disbursement Amount Over Time with a better color
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Disbursement Amount Over Time")
        disbursement_over_time = filtered_data.groupby('Disbursement_Date')['Disbursement_Amount'].sum().reset_index()
        fig_line_chart = px.line(
            disbursement_over_time,
            x='Disbursement_Date',
            y='Disbursement_Amount',
            title='Total Disbursement Amount Over Time',
            labels={'Disbursement_Date': 'Date', 'Disbursement_Amount': 'Disbursement Amount'},
            line_shape='linear',
            markers=False
        )
        fig_line_chart.update_traces(line=dict(color='#2ca02c')) # A professional green
        fig_line_chart.update_layout(hovermode="x unified")
        st.plotly_chart(fig_line_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col_chart3, col_chart4 = st.columns(2)

        with col_chart3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Recovered Amount vs. Service Charge")
            pie_data = pd.DataFrame({
                'Category': ['Recovered Amount', 'Service Charge'],
                'Amount': [total_recovered, total_service_charge]
            })
            fig_recovered_service_pie = px.pie(
                pie_data,
                values='Amount',
                names='Category',
                title='Proportion of Recovered Amount to Service Charge',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_recovered_service_pie.update_traces(textinfo='percent+value', texttemplate='%{percent}<br>%{value:,.0f}', textfont_size=16)
            st.plotly_chart(fig_recovered_service_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Recovered Amount by Loan Cycle")
            recovered_by_cycle = filtered_data.groupby('Loan_Cycle')['Recovered_Amount'].sum().reset_index()
            fig_recovered_bar = px.bar(
                recovered_by_cycle,
                x='Loan_Cycle',
                y='Recovered_Amount',
                title='Total Recovered Amount by Loan Cycle',
                labels={'Loan_Cycle': 'Loan Cycle', 'Recovered_Amount': 'Recovered Amount'},
                text='Recovered_Amount',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_recovered_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside', textfont_size=14)
            st.plotly_chart(fig_recovered_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Top 60 Borrowers by Disbursement Plot with improved colors ---
        st.header("Top 60 Borrowers by Disbursement")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        top_borrowers = filtered_data.groupby('Borrower_Code')['Disbursement_Amount'].sum().nlargest(60).reset_index()
        top_borrowers.rename(columns={'Disbursement_Amount': 'Total_Disbursement_Amount'}, inplace=True)

        fig_top_borrowers = px.bar(
            top_borrowers.sort_values('Total_Disbursement_Amount', ascending=True),
            y='Borrower_Code',
            x='Total_Disbursement_Amount',
            orientation='h',
            title='Top 60 Borrowers by Total Disbursement Amount',
            labels={'Borrower_Code': 'Borrower Code', 'Total_Disbursement_Amount': 'Total Disbursement Amount'},
            text='Total_Disbursement_Amount',
            color='Total_Disbursement_Amount', # Using a sequential color scale
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_top_borrowers.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_top_borrowers.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title=None,
            height=900
        )
        st.plotly_chart(fig_top_borrowers, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Top 60 Borrowers by Recovered Amount Plot with improved colors ---
        st.header("Top 60 Borrowers by Recovered Amount")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        top_recovered_borrowers = filtered_data.groupby('Borrower_Code')['Recovered_Amount'].sum().nlargest(60).reset_index()
        top_recovered_borrowers.rename(columns={'Recovered_Amount': 'Total_Recovered_Amount'}, inplace=True)

        fig_top_recovered_borrowers = px.bar(
            top_recovered_borrowers.sort_values('Total_Recovered_Amount', ascending=True),
            y='Borrower_Code',
            x='Total_Recovered_Amount',
            orientation='h',
            title='Top 60 Borrowers by Total Recovered Amount',
            labels={'Borrower_Code': 'Borrower Code', 'Total_Recovered_Amount': 'Total Recovered Amount'},
            text='Total_Recovered_Amount',
            color='Total_Recovered_Amount', # Using a different sequential color scale
            color_continuous_scale=px.colors.sequential.Plasma
        )
        fig_top_recovered_borrowers.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_top_recovered_borrowers.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title=None,
            height=900
        )
        st.plotly_chart(fig_top_recovered_borrowers, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")

        # --- Raw Data Section ---
        with st.expander("View Filtered Raw Data"):
            st.dataframe(filtered_data)
        
        # --- TIME-SERIES FORECASTING WITH PROPHET ---
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.header("Recovered Amount Forecast for the Next 3 Months")

        # Prepare data for Prophet: 'ds' for date, 'y' for value
        recovered_df = filtered_data[['Disbursement_Date', 'Recovered_Amount']].copy()
        recovered_df.columns = ['ds', 'y']
        
        # Aggregate by week to smooth out the data
        recovered_df['ds'] = pd.to_datetime(recovered_df['ds'])
        recovered_df = recovered_df.groupby(pd.Grouper(key='ds', freq='W-MON')).sum().reset_index()

        # Initialize and fit the Prophet model
        m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(recovered_df)

        # Create a DataFrame for future dates (3 months)
        future = m.make_future_dataframe(periods=90)
        
        # Make predictions
        forecast = m.predict(future)
        
        # Plot the forecast
        fig_forecast = go.Figure()
        
        # Add actual historical data with a new color
        fig_forecast.add_trace(go.Scatter(
            x=recovered_df['ds'], 
            y=recovered_df['y'], 
            mode='lines+markers', 
            name='Actual Recovered Amount',
            line=dict(color='#2ca02c') # Green
        ))
        
        # Add the forecast line with a new color
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'], 
            mode='lines', 
            name='Forecasted Recovered Amount',
            line=dict(color='#ff7f0e', width=2) # Orange
        ))
        
        # Add the uncertainty interval
        fig_forecast.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'].iloc[::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)', # Lighter orange
            line=dict(color='rgba(255,255,255,0)'),
            name='Uncertainty Interval'
        ))
        
        fig_forecast.update_layout(
            title="Recovered Amount Forecast",
            xaxis_title="Date",
            yaxis_title="Recovered Amount",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Get the last forecasted date and value
        last_forecast_date = forecast['ds'].iloc[-1].strftime('%B %d, %Y')
        last_forecast_value = forecast['yhat'].iloc[-1]
        
        # Add a final line with the last predicted amount
        st.markdown(
            f"**Final Forecast:** The predicted recovered amount for **{last_forecast_date}** is **{last_forecast_value:,.2f}**."
        )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Please adjust the filters to view the dashboard.")
