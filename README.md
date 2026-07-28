# MediAgent AI

MediAgent AI is a Streamlit-based medical report interpreter. It uses an OCR agent to extract lab values from PDF reports and a RAG agent to analyze the results against a small medical knowledge base.

## Features

- Upload a PDF medical lab report
- Extract test results from the report
- Compare findings with medical context
- Show a final medical summary in the app

## Project Structure

- `app.py` - Streamlit UI entrypoint
- `agents/ocr_agent.py` - PDF text extraction and JSON parsing
- `agents/rag_agent.py` - Medical risk analysis with retrieval-augmented generation
- `agents/workflow.py` - LangGraph workflow that connects the agents
- `core/config.py` - LLM and embeddings configuration
- `data/` - Local data files such as sample PDFs
- `vector_db/` - Generated ChromaDB storage

## Requirements

Install the Python packages listed in `requirements.txt`.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key to a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the app

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

## Notes

- `vector_db/` is generated locally and is ignored by Git.
- The current UI and report flow are set to English output.
