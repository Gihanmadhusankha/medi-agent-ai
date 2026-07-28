import os
import streamlit as st
import tempfile

# Add the agents folder to the path so workflow can be imported
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))
from workflow import app as agent_workflow

# Streamlit Page Configuration
st.set_page_config(
    page_title="MediAgent: AI Medical Report Interpreter",
    page_icon="🩺",
    layout="centered"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0284c7;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# UI Header
st.markdown('<p class="main-header">🩺 MediAgent AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Smart Medical Report Interpreter & Health Advisory System</p>', unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/experimental-idea--v2.png", width=80)
    st.subheader("About MediAgent")
    st.info(
        "This is an **Agentic AI** system powered by LangGraph and Groq (Llama 3.1). "
        "It automatically extracts lab values, checks them against a medical knowledge base, "
        "and flags critical health concerns safely."
    )
    st.warning("⚠️ **Disclaimer:** This AI is for informational purposes only and does not replace professional medical diagnosis.")

# Main Upload Section
output_language = "English"
st.caption("Output language: English")

uploaded_file = st.file_uploader("Upload your Medical Lab Report (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Display file details
    file_details = {"FileName": uploaded_file.name, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
    st.write("📄 **Uploaded File Details:**")
    st.json(file_details)
    
    if st.button("🚀 Analyze Report with Agents", type="primary"):
        # Temporarily save the uploaded PDF to disk so pdfplumber can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
            
        try:
            # Progress bar and status updates
            with st.status("🤖 Agents are working on your report...", expanded=True) as status:
                st.write("🔍 **[Agent 1: OCR]** Reading PDF and extracting lab values...")
                
                # Run the LangGraph workflow
                inputs = {"pdf_path": tmp_path, "output_language": "English"}
                result = agent_workflow.invoke(inputs)
                st.write("🧠 **[Agent 2: RAG]** Analyzing health risks against medical knowledge base...")
                st.write("✅ Analysis completed successfully!")
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # Display Final Report
            st.markdown("---")
            st.subheader("🩺 AI Medical Analysis Report")
            st.markdown(result["final_report"])
            
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)