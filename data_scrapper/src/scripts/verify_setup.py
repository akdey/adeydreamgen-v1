import os
import requests
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

def verify():
    print("--- Verifying Setup ---")
    
    # 1. Check Pexels Key
    pexels_key = os.getenv("PEXELS_API_KEY")
    if pexels_key:
        print("✅ PEXELS_API_KEY found.")
        # Optional: Test call
        res = requests.get("https://api.pexels.com/v1/videos/search?query=test&per_page=1", headers={"Authorization": pexels_key})
        if res.status_code == 200:
            print("✅ Pexels API connection successful.")
        else:
            print(f"❌ Pexels API test failed: {res.status_code}")
    else:
        print("❌ PEXELS_API_KEY missing.")

    # 2. Check HF Token & Repo
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO_ID")
    if hf_token and hf_repo:
        print(f"✅ HF_TOKEN and HF_REPO_ID ({hf_repo}) found.")
        try:
            api = HfApi(token=hf_token)
            api.repo_info(repo_id=hf_repo, repo_type="dataset")
            print("✅ Hugging Face repo access verified.")
        except Exception as e:
            print(f"❌ Hugging Face verification failed: {e}")
    else:
        print("❌ HF_TOKEN or HF_REPO_ID missing.")

if __name__ == "__main__":
    verify()
