import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Loan Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for Styling the Dashboard ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 35px;
        padding-bottom: 35px;
    }
    .kpi-container, .chart-card, .analysis-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    .kpi-box {
        background: linear-gradient(135deg, #e6f3ff, #e0e8f0);
        border: 1px solid #c8d9e6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    .kpi-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .kpi-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c5282;
    }
    .chart-label {
        font-weight: bold;
    }
    h1, h2, h3, h4 {
        color: #333;
    }
    p {
        color: #555;
    }
    .main-header {
        text-align: center;
        margin-bottom: 30px;
        color: #1e88e5;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load and preprocess data ---
@st.cache_data
def load_and_clean_data(file_name):
    """
    Loads the loan data from a CSV file, cleans it, and prepares it for analysis.
    """
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_name}' was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

    # Handle numeric columns with commas and convert to float
    for col in ['Disbursement_Amount', 'Service_Charge', 'Repay_Amount', 'Recovered_Amount', 'End_Balance']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Convert 'Disbursement_Date' to a proper datetime object
    df['Disbursement_Date'] = pd.to_datetime(df['Disbursement_Date'], format='%d-%m-%y')
    
    # Fill any remaining missing values with 0
    df = df.fillna(0)

    return df

# The filename is set here to match the file you provided.
file_name = 'Dhaka_0043_Loan_Disbursment_Recovery_Jan_2025_Aug_2025.csv'
df = load_and_clean_data(file_name)

if df is not None:
    # --- Dashboard Title and Overall KPIs (The "Cover Box") ---
    st.markdown("<h1 class='main-header'>Dhaka Branch Loan Disbursment-Recovery Jan 2025 Aug 2025 Performance Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='kpi-container'>", unsafe_allow_html=True)
    st.subheader("Overall Key Metrics")

    # Calculate KPIs on the entire dataset
    total_disbursement = df['Disbursement_Amount'].sum()
    total_recovery = df['Recovered_Amount'].sum()
    total_repay_amount = df['Repay_Amount'].sum()
    total_service_charge = df['Service_Charge'].sum()
    total_end_balance = df['End_Balance'].sum()
    total_borrowers = df['Borrower_Code'].nunique()
    
    # Display KPIs in styled columns without commas or dollar signs
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total Disbursement</div>
            <div class="kpi-value"><b>{total_disbursement:.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total Recovery</div>
            <div class="kpi-value"><b>{total_recovery:.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total Repay Amount</div>
            <div class="kpi-value"><b>{total_repay_amount:.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total Service Charge</div>
            <div class="kpi-value"><b>{total_service_charge:.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total End Balance</div>
            <div class="kpi-value"><b>{total_end_balance:.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Total Borrowers</div>
            <div class="kpi-value"><b>{total_borrowers}</b></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Sidebar for filtering ---
    st.sidebar.header("Filter Options")
    st.sidebar.markdown("Use the options below to filter the charts and raw data.")

    # Multiselect filter for Loan_product
    selected_loan_products = st.sidebar.multiselect(
        "Select Loan Product(s)",
        options=df['Loan_product'].unique(),
        default=df['Loan_product'].unique()
    )

    # Multiselect filter for Loan_cycle
    selected_loan_cycles = st.sidebar.multiselect(
        "Select Loan Cycle(s)",
        options=df['Loan_cycle'].unique(),
        default=df['Loan_cycle'].unique()
    )

    # Date range slider
    min_date = df['Disbursement_Date'].min().date()
    max_date = df['Disbursement_Date'].max().date()
    date_range = st.sidebar.date_input(
        "Select a Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Check if the date range is valid before proceeding
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Filter the dataframe based on user selections
        filtered_df = df[
            df['Loan_product'].isin(selected_loan_products) & 
            df['Loan_cycle'].isin(selected_loan_cycles) &
            (df['Disbursement_Date'].dt.date >= start_date) &
            (df['Disbursement_Date'].dt.date <= end_date)
        ]
    else:
        st.warning("Please select a valid date range.")
        filtered_df = df[
            df['Loan_product'].isin(selected_loan_products) & 
            df['Loan_cycle'].isin(selected_loan_cycles)
        ]

    # --- Interactive Charts ---
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.header("Interactive Data Visualizations")

    # Group data by Disbursement_Date for time-series charts
    df_by_date = filtered_df.groupby('Disbursement_Date').agg(
        Total_Disbursement=('Disbursement_Amount', 'sum'),
        Total_Recovery=('Recovered_Amount', 'sum'),
        Total_End_Balance=('End_Balance', 'sum')
    ).reset_index()

    # Line chart for Disbursement and Recovery over time with new style
    st.subheader("Disbursement vs. Recovery Over Time")
    fig_line_disbursement = go.Figure()
    fig_line_disbursement.add_trace(go.Scatter(
        x=df_by_date['Disbursement_Date'],
        y=df_by_date['Total_Disbursement'],
        mode='lines',
        name='Total Disbursement',
        line=dict(color='#3498db', width=2)
    ))
    fig_line_disbursement.add_trace(go.Scatter(
        x=df_by_date['Disbursement_Date'],
        y=df_by_date['Total_Recovery'],
        mode='lines',
        name='Total Recovery',
        line=dict(color='#2ecc71', width=2)
    ))
    fig_line_disbursement.update_layout(
        title_text="Disbursement and Recovery Trends",
        height=400,
        xaxis_title="Date",
        yaxis_title="Amount",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line_disbursement, use_container_width=True)
    
    # New Line chart for Recovery vs. End Balance over time with new style
    st.subheader("Recovery vs. End Balance Over Time")
    fig_line_balance = go.Figure()
    fig_line_balance.add_trace(go.Scatter(
        x=df_by_date['Disbursement_Date'],
        y=df_by_date['Total_Recovery'],
        mode='lines',
        name='Total Recovery',
        line=dict(color='#2ecc71', width=2)
    ))
    fig_line_balance.add_trace(go.Scatter(
        x=df_by_date['Disbursement_Date'],
        y=df_by_date['Total_End_Balance'],
        mode='lines',
        name='Total End Balance',
        line=dict(color='#f1c40f', width=2)
    ))
    fig_line_balance.update_layout(
        title_text="Recovery and End Balance Trends",
        height=400,
        xaxis_title="Date",
        yaxis_title="Amount",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line_balance, use_container_width=True)

    # Group data by Loan_product for bar chart
    df_by_product = filtered_df.groupby('Loan_product').agg(
        Total_Disbursement=('Disbursement_Amount', 'sum'),
        Total_Recovery=('Recovered_Amount', 'sum')
    ).reset_index()
    
    # Bar chart for Disbursement and Recovery by Loan Product with bold values
    st.subheader("Disbursement vs. Recovery by Loan Product")
    fig_bar_product = px.bar(
        df_by_product,
        x='Loan_product',
        y=['Total_Disbursement', 'Total_Recovery'],
        labels={
            "value": "Amount", 
            "variable": "Metric",
            "Loan_product": "Loan Product"
        },
        title="Performance by Loan Product",
        barmode='group',
        height=400
    )
    fig_bar_product.update_traces(texttemplate='<b>%{y:,.0f}</b>', textposition='outside')
    fig_bar_product.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_bar_product, use_container_width=True)

    # Group data by Loan_cycle for new bar charts
    df_by_cycle = filtered_df.groupby('Loan_cycle').agg(
        Total_Disbursement=('Disbursement_Amount', 'sum'),
        Total_Recovery=('Recovered_Amount', 'sum'),
        Total_End_Balance=('End_Balance', 'sum')
    ).reset_index().sort_values('Loan_cycle')
    
    # New Bar chart for Disbursement by Loan Cycle with bold values
    st.subheader("Disbursement by Loan Cycle")
    fig_bar_disbursement_cycle = px.bar(
        df_by_cycle,
        x='Loan_cycle',
        y='Total_Disbursement',
        labels={
            "Total_Disbursement": "Total Disbursement Amount", 
            "Loan_cycle": "Loan Cycle"
        },
        title="Total Disbursement by Loan Cycle",
        height=400
    )
    fig_bar_disbursement_cycle.update_traces(texttemplate='<b>%{y:,.0f}</b>', textposition='outside')
    fig_bar_disbursement_cycle.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_bar_disbursement_cycle, use_container_width=True)

    # New Bar chart for End Balance by Loan Cycle with bold values
    st.subheader("End Balance by Loan Cycle")
    fig_bar_end_balance_cycle = px.bar(
        df_by_cycle,
        x='Loan_cycle',
        y='Total_End_Balance',
        labels={
            "Total_End_Balance": "Total End Balance Amount", 
            "Loan_cycle": "Loan Cycle"
        },
        title="Total End Balance by Loan Cycle",
        height=400
    )
    fig_bar_end_balance_cycle.update_traces(texttemplate='<b>%{y:,.0f}</b>', textposition='outside')
    fig_bar_end_balance_cycle.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_bar_end_balance_cycle, use_container_width=True)

    # Bar chart for Recovery by Loan Cycle with bold values
    st.subheader("Recovery by Loan Cycle")
    fig_bar_recovery_cycle = px.bar(
        df_by_cycle,
        x='Loan_cycle',
        y='Total_Recovery',
        labels={
            "Total_Recovery": "Total Recovered Amount", 
            "Loan_cycle": "Loan Cycle"
        },
        title="Total Recovery by Loan Cycle",
        height=400
    )
    fig_bar_recovery_cycle.update_traces(texttemplate='<b>%{y:,.0f}</b>', textposition='outside')
    fig_bar_recovery_cycle.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_bar_recovery_cycle, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Key Insights Section ---
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.header("Key Insights")

    # Set the number of top borrowers to display
    TOP_N = 15

    # Top 15 Borrowers by Recovery
    st.subheader("Top 15 Borrower_Code by Total Recovery (Out of 1206 Borrowers)")
    top_recovery_borrowers = filtered_df.groupby('Borrower_Code')['Recovered_Amount'].sum().sort_values(ascending=False).head(TOP_N).reset_index()
    fig_top_recovery = px.bar(
        top_recovery_borrowers,
        x='Recovered_Amount',
        y='Borrower_Code',
        orientation='h',
        title=f'Top {TOP_N} Borrower_Code by Total Recovery Amount'
    )
    st.plotly_chart(fig_top_recovery, use_container_width=True)
    
    # Top 15 Borrowers by Disbursement
    st.subheader("Top 15 Borrower_Code by Total Disbursement (Out of 1206 Borrowers)")
    top_disbursement_borrowers = filtered_df.groupby('Borrower_Code')['Disbursement_Amount'].sum().sort_values(ascending=False).head(TOP_N).reset_index()
    fig_top_disbursement = px.bar(
        top_disbursement_borrowers,
        x='Disbursement_Amount',
        y='Borrower_Code',
        orientation='h',
        title=f'Top {TOP_N} Borrower_Code by Total Disbursement Amount'
    )
    st.plotly_chart(fig_top_disbursement, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Machine Learning Analysis for Recovery ---
    st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
    st.header("Best Recovery Prediction Analysis")
    st.markdown("""
        To predict which loans are most likely to have a high recovery, we will use a **Random Forest Classifier** and **Logistic Regression**.
        
        For this analysis, we define a "best recovery" as a `Recovered_Amount` that is greater than the 75th percentile of all recovery values in the dataset.
    """)

    # 1. Prepare the data for the model
    # Define features and target
    features = ['Disbursement_Amount', 'Service_Charge', 'Loan_cycle', 'Repay_Amount']
    
    # Calculate the 75th percentile of Recovered_Amount to define "best recovery"
    recovery_threshold = df['Recovered_Amount'].quantile(0.75)
    df['Is_Best_Recovery'] = (df['Recovered_Amount'] > recovery_threshold).astype(int) # 1 for True, 0 for False

    # Prepare features (X) and target (y)
    X = df[features]
    y = df['Is_Best_Recovery']
    
    # Check for empty data before proceeding
    if not X.empty:
        # 2. Split data and train the models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the Random Forest Classifier
        st.subheader("Random Forest Classifier Results")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions and evaluate the Random Forest model
        y_pred_rf = rf_model.predict(X_test)
        
        # Calculate accuracy and generate a classification report
        model_accuracy_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf, target_names=['Not Best Recovery', 'Best Recovery'], output_dict=True)

        st.info(f"The **Random Forest** model's overall accuracy is: **{model_accuracy_rf:.2%}**")
        st.markdown("#### Detailed Classification Report (Random Forest)")
        st.table(pd.DataFrame(report_rf).transpose().style.format(precision=2))

        # Visualize Feature Importance
        st.markdown("#### Feature Importance Analysis (Random Forest)")
        feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
        
        fig_rf, ax_rf = plt.subplots(figsize=(14, 4))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax_rf, palette="viridis")
        ax_rf.set_title("Feature Importance")
        ax_rf.set_xlabel("Importance Score")
        ax_rf.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig_rf) 
        
        st.markdown("""
            The Random Forest classifier highlights which features are the strongest predictors of a successful recovery, with higher scores indicating more importance.
        """)
        
        # --- New Logistic Regression Section ---
        st.subheader("Logistic Regression Results")
        lr_model = LogisticRegression(random_state=42, max_iter=200)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, y_pred_lr)
        
        st.info(f"The **Logistic Regression** model's overall accuracy is: **{lr_accuracy:.2%}**")
        
        # Display detailed classification report
        report_lr = classification_report(y_test, y_pred_lr, target_names=['Not Best Recovery', 'Best Recovery'], output_dict=True)
        st.markdown("#### Detailed Classification Report (Logistic Regression)")
        st.table(pd.DataFrame(report_lr).transpose().style.format(precision=2))

        # Add confusion matrix for Logistic Regression
        st.markdown("#### Confusion Matrix (Logistic Regression)")
        st.markdown("""
            A confusion matrix shows how well the model's predictions match the actual values.
            - **True Positives (TP):** Correctly predicted 'Best Recovery'.
            - **True Negatives (TN):** Correctly predicted 'Not Best Recovery'.
            - **False Positives (FP):** Incorrectly predicted 'Best Recovery'.
            - **False Negatives (FN):** Incorrectly predicted 'Not Best Recovery'.
        """)
        cm = confusion_matrix(y_test, y_pred_lr)
        fig_cm, ax_cm = plt.subplots(figsize=(14, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Best', 'Best'], yticklabels=['Not Best', 'Best'])
        ax_cm.set_ylabel('Actual Label')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        
        # 5. Model Prediction Visualization with Top Features
        st.subheader("Model Prediction Visualization")
        st.markdown("""
        This interactive plot shows the model's predictions using the two most important features it identified.
        The color represents the model's predicted outcome, allowing you to see how the most influential variables separate "Best Recovery" loans from others.
        """)
        
        # Get the top 2 most important features from the feature_importances series
        top_two_features = feature_importances.index[:2].tolist()
        
        if len(top_two_features) >= 2:
            x_feature = top_two_features[0]
            y_feature = top_two_features[1]
            
            # Add predictions to the test dataframe for plotting
            X_test_plot = X_test.copy()
            X_test_plot['Predicted_Recovery'] = np.where(y_pred_rf == 1, 'Best Recovery', 'Not Best Recovery')
            
            # Create the new interactive scatter plot with a new color map
            fig_pred_scatter = px.scatter(
                X_test_plot,
                x=x_feature,
                y=y_feature,
                color='Predicted_Recovery',
                color_discrete_map={'Best Recovery': '#27ae60', 'Not Best Recovery': '#e74c3c'},
                title=f'Prediction of Best Recovery by {x_feature} and {y_feature}',
                hover_data=X_test.columns,
                labels={
                    x_feature: x_feature.replace('_', ' '),
                    y_feature: y_feature.replace('_', ' ')
                }
            )
            st.plotly_chart(fig_pred_scatter, use_container_width=True)

            # Add a new bar chart to show the distribution of predicted outcomes
            st.subheader("Distribution of Predicted Outcomes")
            predicted_counts = X_test_plot['Predicted_Recovery'].value_counts().reset_index()
            predicted_counts.columns = ['Predicted Outcome', 'Count']
            
            fig_bar_pred = px.bar(
                predicted_counts,
                x='Predicted Outcome',
                y='Count',
                color='Predicted Outcome',
                color_discrete_map={'Best Recovery': '#27ae60', 'Not Best Recovery': '#e74c3c'},
                title="Count of Predicted 'Best Recovery' Loans",
                labels={'Count': 'Number of Loans'}
            )
            fig_bar_pred.update_layout(
                xaxis_title="Predicted Outcome",
                yaxis_title="Count of Loans"
            )
            fig_bar_pred.update_traces(texttemplate='<b>%{y}</b>', textposition='outside')
            st.plotly_chart(fig_bar_pred, use_container_width=True)

        else:
            st.warning("Not enough features available to create the prediction visualization with the top two features.")

    else:
        st.warning("No data available for the selected filters to perform the analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Raw Data Table ---
    st.markdown("---")
    st.subheader("Filtered Raw Data")
    st.dataframe(filtered_df)

else:
    st.error("Please ensure the CSV file is correctly named and uploaded to the same directory as the app.py file.")
