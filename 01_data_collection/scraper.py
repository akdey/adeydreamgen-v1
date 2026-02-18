"""
01_data_collection/scraper.py

Dual-Source Video Scraper (Pexels + Pixabay)
Downloads high-quality cinematic videos and uploads them to Hugging Face Dataset.
"""
import os
import requests
import time
import json
import random
from huggingface_hub import HfApi, create_repo

# --- CONFIGURATION ---
DATASET_REPO = os.getenv("HF_REPO_ID", "a-k-dey/akd-video-training-dataset")
HF_TOKEN = os.getenv("HF_TOKEN")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_KEY = os.getenv("PIXABAY_API_KEY")

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Search Queries (Cinematic focus)
QUERIES = [
    "cinematic nature drone 4k",
    "misty forest cinematic",
    "ocean waves slow motion",
    "cyberpunk city rain",
    "futuristic technology 3d render",
    "macro slow motion coffee",
    "fire crackling low light",
    "desert aerial sunset",
    "snow falling slow motion",
    "underwater coral reef 4k"
]

def search_pexels(query, per_page=15):
    if not PEXELS_KEY: return []
    headers = {"Authorization": PEXELS_KEY}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}&orientation=landscape"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.json().get("videos", [])
        print(f"Pexels Error: {r.status_code}")
    except Exception as e:
        print(f"Pexels Exception: {e}")
    return []

def search_pixabay(query, per_page=15):
    if not PIXABAY_KEY: return []
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_KEY}&q={query}&per_page={per_page}&video_type=film"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json().get("hits", [])
        print(f"Pixabay Error: {r.status_code}")
    except Exception as e:
        print(f"Pixabay Exception: {e}")
    return []

def download_video(url, filename):
    if os.path.exists(filename): return True
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        return False

def main():
    api = HfApi()

    
    # ---------------------------------------------------------
    # 🧭 Intelligent Exploration Mode
    # Instead of looping the same keywords, we "drift" to new related topics.
    # "Forest" -> "Mist" -> "Witcher" -> "Medieval"...
    # ---------------------------------------------------------
    
    # Check if we left a breadcrumb last time
    if os.path.exists("last_query.txt"):
        with open("last_query.txt", "r") as f:
            query = f.read().strip()
    else:
        # Start fresh
        query = random.choice(QUERIES)
        
    print(f"🎨 Starting Exploration: '{query}'")
    
    # 2. Fetch Candidates
    candidates = []
    
    # Pexels
    pex_vids = search_pexels(query, 10)
    for v in pex_vids:
        # Find best quality MP4 (HD but not huge 4K file if possible, or 4K if available)
        files = v.get("video_files", [])
        # Sort by resolution desc
        files.sort(key=lambda x: x["width"] * x["height"], reverse=True)
        best = next((f for f in files if f["width"] >= 1280), files[0] if files else None)
        if best:
            tags = v["url"].split("/")[-2].replace("-", ", ")
            candidates.append({
                "source": "pexels",
                "id": str(v["id"]),
                "url": best["link"],
                "tags": tags,
                "desc": f"Pexels video {v['id']}"
            })
            
    # Pixabay
    pix_vids = search_pixabay(query, 10)
    for v in pix_vids:
        best = v.get("videos", {}).get("large", {}).get("url") or v.get("videos", {}).get("medium", {}).get("url")
        if best:
            candidates.append({
                "source": "pixabay",
                "id": str(v["id"]),
                "url": best,
                "tags": v["tags"],
                "desc": f"Pixabay video {v['id']}"
            })
    
    print(f"Found {len(candidates)} candidates.")
    
    # 3. Download & Upload
    new_topics = []
    
    for vid in candidates:
        vid_filename = f"{vid['source']}_{vid['id']}.mp4"
        txt_filename = f"{vid['source']}_{vid['id']}.txt"
        
        local_vid = os.path.join(VIDEO_DIR, vid_filename)
        local_txt = os.path.join(VIDEO_DIR, txt_filename)
        
        # Collect tags for next jump
        if vid["tags"]:
            new_topics.extend([t.strip() for t in vid["tags"].split(",") if len(t.strip()) > 3])
        
        # Download
        # Check if file already exists in HF dataset (Optional optimization, complicated API calls so skip)
        
        print(f"⬇️ Downloading {vid_filename}...")
        if download_video(vid["url"], local_vid):
            # Create Caption
            caption = f"cinematic video of {vid['tags']}. {vid['desc']}"
            with open(local_txt, "w") as f:
                f.write(caption)
                
            # Upload to HF
            print(f"⬆️ Uploading to {DATASET_REPO}...")
            try:
                # Upload Video
                api.upload_file(
                    path_or_fileobj=local_vid,
                    path_in_repo=vid_filename,
                    repo_id=DATASET_REPO,
                    repo_type="dataset"
                )
                # Upload Caption
                api.upload_file(
                    path_or_fileobj=local_txt,
                    path_in_repo=txt_filename,
                    repo_id=DATASET_REPO,
                    repo_type="dataset"
                )
                print("✅ Uploaded!")
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                
            # Clean up to save space
            if os.path.exists(local_vid): os.remove(local_vid)
            if os.path.exists(local_txt): os.remove(local_txt)

    # 4. Drift to New Topic
    if new_topics:
        next_query = random.choice(new_topics)
        print(f"🌊 Drifting to new topic: '{next_query}'")
        with open("last_query.txt", "w") as f:
            f.write(next_query)
    else:
        print("⚓ No new topics found. Staying on current course.")

if __name__ == "__main__":
    main()
