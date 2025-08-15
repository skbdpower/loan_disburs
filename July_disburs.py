import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from PIL import Image

# Suppress the FutureWarning from Plotly
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(
    page_title="Loan Disbursement Dashboard",
    layout="wide"
)

# Load the TMSS logo
try:
    tmss_logo = Image.open('tmss.jpg')  # Updated to look for the .jpg file
except FileNotFoundError:
    st.error("Error: 'tmss.jpg' not found. Please ensure the image file is in the same directory as the script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the logo: {e}")
    st.stop()

# Load the data from the CSV file
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("All_OP_july_2025_Disburs.csv")
        # Ensure numerical columns are of the correct type
        data['Loan_Amount'] = pd.to_numeric(data['Loan_Amount'].str.replace(',', ''), errors='coerce')
        data['Outstanding_Pr'] = pd.to_numeric(data['Outstanding_Pr'].str.replace(',', ''), errors='coerce')
        return data
    except FileNotFoundError:
        st.error("Error: 'All_OP_july_2025_Disburs.csv' not found. Please ensure the CSV file is in the same directory as the script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

# Main dashboard title
st.title("💸 July 2025 Loan Disbursement Analysis Dashboard")
st.markdown("An interactive dashboard to visualize and analyze loan data from the provided dataset.")

df = load_data()

# Create a sidebar for filters
st.sidebar.image(tmss_logo, use_container_width=True) # Updated to use use_container_width
st.sidebar.header("Filter Options")
all_divisions = ['All'] + list(df['Divisional_Office'].unique())
selected_division = st.sidebar.selectbox("Select Divisional Office", all_divisions)

if selected_division != 'All':
    filtered_df = df[df['Divisional_Office'] == selected_division]
    all_zones = ['All'] + list(filtered_df['Zone_Office'].unique())
    selected_zone = st.sidebar.selectbox("Select Zone Office", all_zones)
    if selected_zone != 'All':
        filtered_df = filtered_df[filtered_df['Zone_Office'] == selected_zone]
        all_areas = ['All'] + list(filtered_df['Area_Office'].unique())
        selected_area = st.sidebar.selectbox("Select Area Office", all_areas)
        if selected_area != 'All':
            filtered_df = filtered_df[filtered_df['Area_Office'] == selected_area]
else:
    filtered_df = df.copy()

# Display key metrics
st.header("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_loan_amount = filtered_df['Loan_Amount'].sum()
total_outstanding_pr = filtered_df['Outstanding_Pr'].sum()
total_loanees = filtered_df['Loanee'].sum()
total_branches = filtered_df['Branch_Name'].nunique()

col1.metric("Total Loan Amount", f"{total_loan_amount:,.0f}")
col2.metric("Total Outstanding Principal", f"{total_outstanding_pr:,.0f}")
col3.metric("Total Loanees", f"{total_loanees:,.0f}")
col4.metric("Total Branches", f"{total_branches:,.0f}")

st.markdown("---")

# Analysis and visualizations
st.header("Data Visualizations")

# Chart 1: Loan Amount by Divisional Office
div_loan_amount = filtered_df.groupby('Divisional_Office')['Loan_Amount'].sum().reset_index()
fig1 = px.bar(
    div_loan_amount,
    x='Divisional_Office',
    y='Loan_Amount',
    title='Total Loan Amount by Divisional Office',
    labels={'Divisional_Office': 'Divisional Office', 'Loan_Amount': 'Total Loan Amount'},
    color='Divisional_Office',
    text_auto=True  # Display values on the bars
)
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Outstanding Principal by Zone Office
zone_outstanding = filtered_df.groupby('Zone_Office')['Outstanding_Pr'].sum().reset_index()
fig2 = px.bar(
    zone_outstanding,
    x='Zone_Office',
    y='Outstanding_Pr',
    title='Total Outstanding Principal by Zone Office',
    labels={'Zone_Office': 'Zone Office', 'Outstanding_Pr': 'Total Outstanding Principal'},
    color='Zone_Office',
    text_auto=True  # Display values on the bars
)
st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Loan Amount vs. Outstanding Principal by Branch
branch_data = filtered_df.groupby('Branch_Name')[['Loan_Amount', 'Outstanding_Pr']].sum().reset_index()
fig3 = px.scatter(
    branch_data,
    x='Loan_Amount',
    y='Outstanding_Pr',
    hover_data=['Branch_Name'],
    title='Loan Amount vs. Outstanding Principal by Branch',
    labels={'Loan_Amount': 'Total Loan Amount', 'Outstanding_Pr': 'Total Outstanding Principal'}
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Display the raw data
st.header("Raw Data Table")
st.dataframe(filtered_df)
