from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Import functions from the agents created earlier
from ocr_agent import extract_text_from_pdf, parse_medical_data
from rag_agent import analyze_health_risks

# 1. Define the shared state between agents
class AgentState(TypedDict):
    pdf_path: str
    output_language: str
    raw_text: str
    extracted_json: List[Dict[str, Any]]
    final_report: str

# 2. First node: OCR agent
def ocr_node(state: AgentState):
    print("\n[Agent 1: OCR] Reading the PDF and extracting data...")
    raw_text = extract_text_from_pdf(state["pdf_path"])
    extracted_json = parse_medical_data(raw_text)
    
    # Return the extracted data to the next agent
    return {"raw_text": raw_text, "extracted_json": extracted_json}

# 3. Second node: RAG analysis agent
def rag_node(state: AgentState):
    print("\n[Agent 2: RAG] Checking the medical knowledge base and analyzing the data...")
    
    # Stop if OCR did not return any data
    if not state.get("extracted_json"):
        return {"final_report": "❌ Error: Could not extract data from the PDF."}
        
    report = analyze_health_risks(
        state["extracted_json"],
        output_language=state.get("output_language", "English"),
    )
    return {"final_report": report}

# 4. Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("ocr", ocr_node)
workflow.add_node("rag", rag_node)

# Define the flow
workflow.set_entry_point("ocr")    # Start with OCR
workflow.add_edge("ocr", "rag")    # Move to RAG after OCR finishes
workflow.add_edge("rag", END)      # End after RAG

# Compile the workflow
app = workflow.compile()

# Test the workflow
if __name__ == "__main__":
    print("🚀 MediAgent AI workflow started!\n")
    
    # Path to the test PDF
    inputs = {"pdf_path": "../data/test.pdf"}
    
    # Run the workflow
    result = app.invoke(inputs)
    
    print("\n" + "="*50)
    print("🩺 Final AI medical report (FINAL REPORT)")
    print("="*50)
    print(result["final_report"])