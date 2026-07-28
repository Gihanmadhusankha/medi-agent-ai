import os
import json
from pathlib import Path
import chromadb
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
# Load functions from the existing config file
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import get_llm, get_embeddings

# ChromaDB storage location
DB_DIR = Path(__file__).resolve().parents[1] / "vector_db" / "chroma_db"


class _ChromadbRetriever:
    def __init__(self, collection, embedding_function, k=2):
        self.collection = collection
        self.embedding_function = embedding_function
        self.k = k

    def invoke(self, query):
        query_embedding = self.embedding_function.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k,
            include=["documents"],
        )
        documents = []
        for page_content in results.get("documents", [[]])[0]:
            documents.append(Document(page_content=page_content))
        return documents


class Chroma:
    def __init__(self, persist_directory, embedding_function, collection_name="medical_kb"):
        self.persist_directory = str(persist_directory)
        self.embedding_function = embedding_function
        client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = client.get_or_create_collection(name=collection_name)

    @classmethod
    def from_documents(cls, documents, embedding, persist_directory):
        store = cls(persist_directory=persist_directory, embedding_function=embedding)
        texts = [document.page_content for document in documents]
        embeddings = embedding.embed_documents(texts)
        ids = [str(index) for index in range(len(texts))]
        store.collection.upsert(ids=ids, documents=texts, embeddings=embeddings)
        return store

    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {}
        return _ChromadbRetriever(
            self.collection,
            self.embedding_function,
            k=search_kwargs.get("k", 2),
        )

def setup_medical_kb():
    """Populate the vector DB with a small medical knowledge base."""
    print("Creating the vector DB...")
    embeddings = get_embeddings()
    
    # Add a few sample medical knowledge entries for simple testing.
    # A larger medical guidelines PDF can be added later.
    sample_knowledge = [
        "High Fasting Blood Sugar (above 106 mg/dL) may indicate prediabetes or diabetes mellitus.",
        "Elevated Triglycerides (above 150 mg/dL) are associated with metabolic syndrome, hypothyroidism, or high carbohydrate diets.",
        "High WBC Count (above 10000 /cmm) usually indicates infection, inflammation, or immune system response.",
        "High Lymphocytes can indicate a viral infection.",
        "Low Hemoglobin indicates anemia, which could be due to iron deficiency or blood loss."
    ]
    
    docs = [Document(page_content=text) for text in sample_knowledge]
    
    # Build and save the DB
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("✅ Medical Knowledge Base setup complete!")
    return vectorstore

def get_vector_store():
    """Load the vector DB if it exists, otherwise create it."""
    embeddings = get_embeddings()
    if not DB_DIR.exists():
        return setup_medical_kb()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def analyze_health_risks(extracted_json_data, output_language="English"):
    """Analyze the extracted JSON and generate a medical summary."""
    vectorstore = get_vector_store()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Convert JSON data to a string so the AI can read it easily.
    findings_str = json.dumps(extracted_json_data, indent=2)

    # Retrieve relevant medical context through RAG.
    # Here we mainly look for blood sugar and triglyceride related context from the DB.
    search_queries = ["Fasting Blood Sugar", "Triglyceride", "WBC Count", "Hemoglobin"]
    context = ""
    for query in search_queries:
        docs = retriever.invoke(query)
        for d in docs:
            context += d.page_content + "\n"

    # System prompt for the agent
    
        language_instructions = """- Provide the final explanation completely in English.
- Explain the medical context in English too.
- Keep the medical test names and units in English so it's easy to understand.
- Keep the explanation simple, clear, and easy for a normal person to understand.
- NEVER diagnose the patient. Always recommend that they consult a doctor.
- If everything is normal, state it clearly in English."""
        user_prompt = "Here are my lab results:\n\n{findings}"
        report_language = "English"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a highly skilled clinical AI assistant.
        Step 1: Look at the patient's lab results provided by the user.
        Step 2: Identify ANY abnormal values by comparing the 'result' with the 'reference_range'.
        Step 3: Use the provided 'Medical Context' to explain what might be causing these abnormal values and if they are risky.

        CRITICAL INSTRUCTIONS:
        {language_instructions}

        Medical Context (source text):
        {{context}}"""),
        ("user", user_prompt)
    ])
    chain = prompt | llm
    
    print("\nSearching the medical knowledge base and analyzing... ⏳")
    response = chain.invoke({"context": context, "findings": findings_str, "report_language": report_language})
    
    return response.content

def chat_with_report(extracted_json_data, user_question, chat_history):
    
    vectorstore = get_vector_store()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    findings_str = json.dumps(extracted_json_data, indent=2)

    docs = retriever.invoke(user_question)
    context = "\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and empathetic clinical AI assistant. 
        You are having a conversation with a user about their medical lab report.
        
        Patient's Lab Results:
        {findings}
        
        Relevant Medical Knowledge:
        {context}
        
        Chat History:
        {chat_history}
        
        CRITICAL INSTRUCTIONS:
        - Answer the user's question clearly and politely in **Sinhala language** (සිංහලෙන්).
        - Keep medical test names and units in English (e.g., Fasting Blood Sugar, mg/dL).
        - Give practical, general lifestyle or dietary tips if asked, but NEVER give a formal medical diagnosis.
        - Always remind them to consult their doctor for medical treatments (වෛද්‍යවරයකු හමුවී උපදෙස් ලබාගන්න).
        """),
        ("user", "{user_question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "findings": findings_str,
        "context": context,
        "chat_history": chat_history,
        "user_question": user_question
    })
    
    return response.content

# Test run
if __name__ == "__main__":
    # This is a dummy JSON extracted from OCR in Phase 1.
    # (We used the report values for Fasting Blood Sugar - 141 and Triglyceride - 168.)
    sample_extracted_data = [
      {
        "test_name": "Fasting Blood Sugar",
        "result": "141.0",
        "unit": "mg/dL",
        "reference_range": "74 - 106"
      },
      {
        "test_name": "Triglyceride",
        "result": "168.0",
        "unit": "mg/dL",
        "reference_range": "<150"
      },
      {
        "test_name": "Hemoglobin",
        "result": "14.5",
        "unit": "g/dL",
        "reference_range": "13.0 - 16.5"
      }
    ]
    
    final_analysis = analyze_health_risks(sample_extracted_data)
    
    print("\n" + "="*50)
    print("🩺 AI MEDICAL ANALYSIS REPORT")
    print("="*50)
    print(final_analysis)