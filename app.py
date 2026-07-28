import os
import streamlit as st
import tempfile

# agents ෆෝල්ඩර් එකෙන් workflows සහ chat function එක import කරගැනීම
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))
from workflow import app as agent_workflow
from rag_agent import chat_with_report

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
    </style>
""", unsafe_allow_html=True)

# UI Header
st.markdown('<p class="main-header">🩺 MediAgent AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Smart Medical Report Interpreter & Interactive Health Assistant</p>', unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/experimental-idea--v2.png", width=80)
    st.subheader("About MediAgent")
    st.info(
        "Powered by **LangGraph, Groq (Llama 3.1) & RAG**. "
        "Upload your report to analyze findings and chat interactively with the AI health assistant."
    )
    st.warning("⚠️ **Disclaimer:** For informational purposes only. Consult a doctor for medical advice.")

# Initialize Session State for maintaining data and chat history across reruns
if "extracted_json" not in st.session_state:
    st.session_state.extracted_json = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Main Upload Section
uploaded_file = st.file_uploader("Upload your Medical Lab Report (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Check if a new file is uploaded to reset previous state if needed
    if st.button("🚀 Analyze Report with Agents", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
            
        try:
            with st.status("🤖 Agents are working on your report...", expanded=True) as status:
                st.write("🔍 **[Agent 1: OCR]** Reading PDF and extracting lab values...")
                
                inputs = {"pdf_path": tmp_path}
                result = agent_workflow.invoke(inputs)
                
                # Save results in session state
                st.session_state.extracted_json = result.get("extracted_json")
                st.session_state.final_report = result.get("final_report")
                # Reset chat history for the new report
                st.session_state.chat_messages = []
                
                st.write("✅ Analysis completed successfully!")
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# Display Final Report if available
if st.session_state.final_report:
    st.markdown("---")
    st.subheader("🩺 AI Medical Analysis Report")
    st.markdown(st.session_state.final_report)
    
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your Report")
    st.write("ඔබේ වාර්තාව පිළිබඳව ඕනෑම ප්‍රශ්නයක් පහතින් අසන්න (උදා: *මේක අඩු කරගන්න කෑම රටාව කොහොමද?*):")

    # Display prior chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    if user_query := st.chat_input("ваша ප්‍රශ්නය මෙහි ලියන්න... / Type your question here..."):
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate AI response using chat history and extracted json context
        with st.chat_message("assistant"):
            with st.spinner("AI එක පිළිතුර සකස් කරමින් පවතී..."):
                # Format history for prompt context
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages[:-1]])
                
                ai_response = chat_with_report(
                    st.session_state.extracted_json, 
                    user_query, 
                    history_str
                )
                st.markdown(ai_response)
                
        # Save assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})