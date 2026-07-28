import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

def _create_huggingface_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ModuleNotFoundError:
        from sentence_transformers import SentenceTransformer

        class SentenceTransformerEmbeddings:
            def __init__(self, model_name):
                self.model = SentenceTransformer(model_name)

            def embed_documents(self, texts):
                return self.model.encode(texts, normalize_embeddings=True).tolist()

            def embed_query(self, text):
                return self.model.encode([text], normalize_embeddings=True)[0].tolist()

        return SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()

def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=2098,
    )

def get_embeddings():
    return _create_huggingface_embeddings()

if __name__ == "__main__":
    llm = get_llm()
    embeddings = get_embeddings()
    response = llm("Hello, What is your role in healthcare?")
    print(f"LLM response: {response}")
    