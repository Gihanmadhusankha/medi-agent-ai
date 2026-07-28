import os
import requests
from dotenv import load_dotenv

# .env එකෙන් API Key එක ගන්නවා
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def get_groq_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Errors තියෙනවද බලනවා
        
        models_data = response.json()
        
        print("✅ ඔයාගේ API Key එකට වැඩ කරන Models List එක:\n")
        for model in models_data.get('data', []):
            print(f"- {model['id']}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error එකක් ආවා: {e}")

if __name__ == "__main__":
    get_groq_models()