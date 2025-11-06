"""
FinCrime-LLM Streamlit Demo Application.

Multi-page demo for SAR generation, KYC assessment, and transaction analysis.
"""

import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="FinCrime-LLM Demo",
    page_icon="=",
    layout="wide",
)

# Sidebar
st.sidebar.title("FinCrime-LLM")
st.sidebar.markdown("AI-powered financial crime detection")

page = st.sidebar.radio(
    "Select Feature",
    ["SAR Generator", "KYC Assessor", "Transaction Analyzer"],
)

API_BASE_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

# SAR Generator Page
if page == "SAR Generator":
    st.title("=¨ SAR Generator")
    st.markdown("Generate Suspicious Activity Reports from transaction data")

    col1, col2 = st.columns(2)

    with col1:
        country = st.text_input("Country", "Ghana")
        subject_name = st.text_input("Subject Name", "John Doe")
        institution = st.text_input("Institution", "First Atlantic Bank")

    with col2:
        total_amount = st.number_input("Total Amount", 100000.0)
        currency = st.text_input("Currency", "GHS")

    transactions = st.text_area(
        "Transaction Details",
        "2024-01-15: 50,000 GHS to Shell Company A\n"
        "2024-01-20: 50,000 GHS to Offshore Account B",
        height=150,
    )

    summary = st.text_area("Summary", "Multiple large transfers to high-risk entities")

    if st.button("Generate SAR", type="primary"):
        with st.spinner("Generating SAR..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/sar/generate",
                    json={
                        "country": country,
                        "subject_name": subject_name,
                        "institution": institution,
                        "total_amount": total_amount,
                        "currency": currency,
                        "transactions": transactions,
                        "summary": summary,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f" SAR Generated: {result['report_id']}")
                    st.markdown("### Generated SAR")
                    st.text_area("SAR Content", result["sar_content"], height=400)

                    # Download button
                    st.download_button(
                        "Download SAR",
                        result["sar_content"],
                        file_name=f"{result['report_id']}.txt",
                    )
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# KYC Assessor Page
elif page == "KYC Assessor":
    st.title("=d KYC Assessor")
    st.markdown("Perform customer due diligence assessments")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Customer Name", "Jane Smith")
        customer_type = st.selectbox("Customer Type", ["Individual", "Entity"])
        country = st.text_input("Country", "Kenya")

    with col2:
        occupation = st.text_input("Occupation/Business", "Import/Export Business")
        source_of_funds = st.text_input("Source of Funds", "Business Revenue")
        expected_volume = st.text_input("Expected Monthly Volume", "100,000 - 500,000 KES")

    additional_info = st.text_area(
        "Additional Information",
        "Customer deals with international suppliers",
    )

    if st.button("Assess KYC Risk", type="primary"):
        with st.spinner("Performing KYC assessment..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/kyc/assess",
                    json={
                        "name": name,
                        "customer_type": customer_type,
                        "country": country,
                        "occupation": occupation,
                        "source_of_funds": source_of_funds,
                        "expected_volume": expected_volume,
                        "additional_info": additional_info,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f" Assessment Complete: {result['assessment_id']}")
                    st.markdown("### KYC Assessment")
                    st.text_area("Assessment", result["kyc_content"], height=400)
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Transaction Analyzer Page
elif page == "Transaction Analyzer":
    st.title("=° Transaction Analyzer")
    st.markdown("Analyze transactions for suspicious patterns")

    transactions = st.text_area(
        "Transaction Data",
        "Account: 123456789\n"
        "2024-01-10: Deposit 10,000 USD\n"
        "2024-01-11: Withdrawal 9,500 USD\n"
        "2024-01-12: Deposit 15,000 USD\n"
        "2024-01-13: Withdrawal 14,500 USD",
        height=200,
    )

    description = st.text_input(
        "Description (optional)",
        "Rapid in-and-out transactions",
    )

    if st.button("Analyze Transactions", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/transaction/analyze",
                    json={
                        "transactions": transactions,
                        "description": description,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f" Analysis Complete: {result['analysis_id']}")
                    st.markdown("### Analysis Results")
                    st.text_area("Analysis", result["analysis_content"], height=400)
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("FinCrime-LLM v1.0.0")
st.sidebar.markdown("[Documentation](https://github.com/yourusername/fincrime-llm)")
