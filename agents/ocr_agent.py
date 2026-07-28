import pdfplumber
import json
import os
import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_llm

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not Found :{pdf_path}")
    text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted=page.extract_text()
            if extracted:
                text+=extracted +"\n"
    return text

def parse_medical_data(raw_text):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert medical data extractor. 
        Extract the laboratory test results from the following text.
        Return ONLY a valid JSON array of objects with keys: 'test_name', 'result', 'unit', and 'reference_range'.
        Do not add any markdown formatting like ```json or extra text. Just the raw JSON array."""),
        ("user", "Here is the raw text from the medical report:\n\n{text}")
    ])
    chain = prompt | llm
    print("Extracting medical data from text...")  # Print first 100 characters for context
    response = chain.invoke({"text": raw_text})
    
    try:
        structured_data = json.loads(response.content.strip())
        return structured_data
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response.content}")
        return None
    
if __name__ == "__main__":
    pdf_path = "../data/test.pdf"  # Replace with your PDF file path
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        print("Extracted Text:\n", raw_text[:500])  # Print first 500 characters for context
        structured_data = parse_medical_data(raw_text)
        if structured_data:
            print("Structured Medical Data:\n", json.dumps(structured_data, indent=2))
        else:
            print("No structured data extracted.")
    except Exception as e:
        print(f"An error occurred: {e}")