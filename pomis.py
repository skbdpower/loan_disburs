import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="POMIS Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("POMIS_Summary.xlsx")
    return df

# Main dashboard
def main():
    st.title("📊 POMIS Microfinance Analytics Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    domains = st.sidebar.multiselect(
        "Select Domain(s)",
        options=df["Domain"].unique(),
        default=df["Domain"].unique()
    )
    
    zones = st.sidebar.multiselect(
        "Select Zone(s)",
        options=df[df["Domain"].isin(domains)]["Zone"].unique(),
        default=df[df["Domain"].isin(domains)]["Zone"].unique()
    )
    
    regions = st.sidebar.multiselect(
        "Select Region(s)",
        options=df[(df["Domain"].isin(domains)) & (df["Zone"].isin(zones))]["Region"].unique(),
        default=df[(df["Domain"].isin(domains)) & (df["Zone"].isin(zones))]["Region"].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df["Domain"].isin(domains)) & 
        (df["Zone"].isin(zones)) & 
        (df["Region"].isin(regions))
    ]
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overview", "Branch Performance", "Financial Analysis", "Trends & Insights"])
    
    with tab1:
        st.header("Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_outstanding = filtered_df["Total Outstanding"].sum()
            st.metric("Total Outstanding", f"{total_outstanding:,.0f}")
        
        with col2:
            total_overdue = filtered_df["Overdue Loan Outstanding"].sum()
            overdue_rate = (total_overdue / total_outstanding * 100) if total_outstanding > 0 else 0
            st.metric("Overdue Rate", f"{overdue_rate:.1f}%")
        
        with col3:
            total_members = filtered_df["New Member Admission (Current Month)"].sum()
            st.metric("New Members", f"{total_members:,}")
        
        with col4:
            recovery_rate = (filtered_df["June_RecoveredRegular"].sum() / 
                           filtered_df["Reg# Loan Recoverable (Current Month)"].sum() * 100) if filtered_df["Reg# Loan Recoverable (Current Month)"].sum() > 0 else 0
            st.metric("Recovery Rate", f"{recovery_rate:.1f}%")
        
        # Portfolio distribution
        col1, col2 = st.columns(2)
        
        with col1:
            domain_summary = filtered_df.groupby("Domain")["Total Outstanding"].sum().reset_index()
            fig = px.pie(domain_summary, values="Total Outstanding", names="Domain", 
                        title="Portfolio Distribution by Domain")
            st.plotly_chart(fig, use_container_width=True)
        
        #with col2:
            #zone_summary = filtered_df.groupby("Zone")["Total Outstanding"].sum().reset_index()
           # fig = px.bar(zone_summary, x="Zone", y="Total Outstanding", 
                        #title="Outstanding Loans by Zone")
           ## fig.update_xaxis(tickangle=45)
           # st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Branch Performance")
        
        # Top performing branches
        col1, col2 = st.columns(2)
        
        with col1:
            top_branches = filtered_df.nlargest(10, "Total Outstanding")[["Branch Name", "Total Outstanding"]]
            fig = px.bar(top_branches, x="Total Outstanding", y="Branch Name", 
                        orientation="h", title="Top 10 Branches by Outstanding Loans")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recovery performance
            filtered_df["Recovery_Rate"] = (filtered_df["June_RecoveredRegular"] / 
                                          filtered_df["Reg# Loan Recoverable (Current Month)"]) * 100
            top_recovery = filtered_df.nlargest(10, "Recovery_Rate")[["Branch Name", "Recovery_Rate"]]
            fig = px.bar(top_recovery, x="Recovery_Rate", y="Branch Name", 
                        orientation="h", title="Top 10 Branches by Recovery Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        # Branch performance table
        st.subheader("Branch Performance Summary")
        performance_summary = filtered_df[[
            "Branch Name", "Domain", "Zone", "Region",
            "Total Outstanding", "Overdue Loan Outstanding", 
            "New Member Admission (Current Month)", "Recovery_Rate"
        ]].round(2)
        st.dataframe(performance_summary, use_container_width=True)
    
    with tab3:
        st.header("Financial Analysis")
        
        # Financial metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Loan disbursement vs collection
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df["Cummulative Loan Disbursement"],
                y=filtered_df["Cummulative Loan Collection"],
                mode="markers",
                text=filtered_df["Branch Name"],
                name="Branches"
            ))
            fig.add_trace(go.Scatter(
                x=[0, filtered_df["Cummulative Loan Disbursement"].max()],
                y=[0, filtered_df["Cummulative Loan Disbursement"].max()],
                mode="lines",
                name="Perfect Collection",
                line=dict(dash="dash")
            ))
            fig.update_layout(
                title="Loan Disbursement vs Collection",
                xaxis_title="Cumulative Disbursement",
                yaxis_title="Cumulative Collection"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bad loan analysis
            filtered_df["Bad_Loan_Rate"] = (filtered_df["Bad Loan(365+)"] / 
                                          filtered_df["Total Outstanding"]) * 100
            fig = px.histogram(filtered_df, x="Bad_Loan_Rate", nbins=20,
                             title="Distribution of Bad Loan Rates")
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional financial performance
        regional_summary = filtered_df.groupby("Region").agg({
            "Total Outstanding": "sum",
            "Overdue Loan Outstanding": "sum",
            "Bad Loan(365+)": "sum",
            "Cummulative Loan Disbursement": "sum",
            "Cummulative Loan Collection": "sum"
        }).reset_index()
        
        regional_summary["Overdue_Rate"] = (regional_summary["Overdue Loan Outstanding"] / 
                                          regional_summary["Total Outstanding"]) * 100
        regional_summary["Collection_Rate"] = (regional_summary["Cummulative Loan Collection"] / 
                                             regional_summary["Cummulative Loan Disbursement"]) * 100
        
        st.subheader("Regional Financial Summary")
        st.dataframe(regional_summary.round(2), use_container_width=True)
    
    with tab4:
        st.header("Trends & Insights")
        
        # Member growth analysis
        col1, col2 = st.columns(2)
        
        with col1:
            member_data = filtered_df.groupby("Domain").agg({
                "New Member Admission (Current Month)": "sum",
                "Member Cancellation (Current Month)": "sum"
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name="New Admissions", x=member_data["Domain"], 
                               y=member_data["New Member Admission (Current Month)"]))
            fig.add_trace(go.Bar(name="Cancellations", x=member_data["Domain"], 
                               y=member_data["Member Cancellation (Current Month)"]))
            fig.update_layout(title="Member Growth by Domain", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Loan performance heatmap
            heatmap_data = filtered_df.pivot_table(
                values="Recovery_Rate", 
                index="Zone", 
                columns="Domain", 
                aggfunc="mean"
            )
            fig = px.imshow(heatmap_data, title="Recovery Rate Heatmap (Zone vs Domain)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = []
        
        # Best performing domain
        domain_performance = filtered_df.groupby("Domain")["Recovery_Rate"].mean()
        best_domain = domain_performance.idxmax()
        insights.append(f"🏆 Best performing domain: {best_domain} with {domain_performance.max():.1f}% recovery rate")
        
        # Highest risk region
        region_risk = filtered_df.groupby("Region")["Bad_Loan_Rate"].mean()
        highest_risk = region_risk.idxmax()
        insights.append(f"⚠️ Highest risk region: {highest_risk} with {region_risk.max():.1f}% bad loan rate")
        
        # Growth leader
        growth_data = filtered_df.groupby("Zone")["New Member Admission (Current Month)"].sum()
        growth_leader = growth_data.idxmax()
        insights.append(f"📈 Growth leader: {growth_leader} with {growth_data.max():,} new members")
        
        for insight in insights:
            st.write(insight)

if __name__ == "__main__":
    main()
