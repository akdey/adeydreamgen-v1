import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pexels_client import PexelsClient
from pixabay_client import PixabayClient
from video_processor import VideoProcessor
from hf_uploader import HFUploader
from dotenv import load_dotenv

load_dotenv()

from categories import CATEGORIES
import random

def main():
    parser = argparse.ArgumentParser(description="ADeyDreamGen-v1 Data Scrapper")
    parser.add_argument("--query", type=str, help="Search query (specific)")
    parser.add_argument("--category_class", type=str, help="Category class from categories.py (e.g. human_activities)")
    parser.add_argument("--items", type=int, default=15, help="Number of items to fetch per source")
    args = parser.parse_args()

    # Determine Queries
    if args.query:
        QUERIES = [args.query]
    elif args.category_class and args.category_class in CATEGORIES:
        # Pick 3 random sub-categories from that class to ensure variety
        QUERIES = random.sample(CATEGORIES[args.category_class], min(3, len(CATEGORIES[args.category_class])))
        print(f"🎲 Randomized class search for [{args.category_class}]: {QUERIES}")
    else:
        # Pick 5 random categories from the entire library
        all_cats = [item for sublist in CATEGORIES.values() for item in sublist]
        QUERIES = random.sample(all_cats, 5)
        print(f"🎲 Total Random exploration: {QUERIES}")

    ITEMS_PER_QUERY = args.items
    
    # Initialize components
    processor = VideoProcessor()
    uploader = HFUploader()
    uploader.cache_repo_files() # Fetch list once to avoid repeated API calls
    
    # Optional Clients
    clients = []
    try:
        clients.append(PexelsClient())
        print("✅ Pexels Client initialized.")
    except Exception as e:
        print(f"⚠️ Pexels Client skipped: {e}")
        
    try:
        clients.append(PixabayClient())
        print("✅ Pixabay Client initialized.")
    except Exception as e:
        print(f"⚠️ Pixabay Client skipped: {e}")

    if not clients:
        print("❌ No scraping clients initialized. Check your .env file.")
        return

    os.makedirs("temp_videos", exist_ok=True)
    
    collected_tags = set()
    session_handled_ids = set()
    
    def process_single_video(v, client, source_name, recurse=True):
        v_id = v['id']
        unique_id = f"{source_name}_{v_id}"

        if unique_id in session_handled_ids:
            return
        session_handled_ids.add(unique_id)
        
        # 🧪 TAG DISCOVERY
        tags = v.get("tags") if isinstance(v.get("tags"), list) else []
        if tags and recurse:
            for t in tags[:3]:
                collected_tags.add(t)

        # Link extraction
        if isinstance(client, PexelsClient):
            best_link = v.get('video_files', [{}])[0].get('link')
            for f in v.get('video_files', []):
                if f['width'] >= 1920 or f['height'] >= 1920:
                    best_link = f['link']
                    break
        else: # Pixabay
            best_link = client.get_best_video_link(v)
        
        if not best_link:
            return

        duration = v.get('duration', 0)
        if duration < 3:
            return
            
        local_name = f"temp_videos/{unique_id}.mp4"
        
        if processor.download_video(best_link, local_name):
            metadata = processor.get_video_metadata(local_name)
            orientation = metadata.get('orientation', 'other')
            
            if orientation in ['landscape', 'portrait']:
                # Deduplication check using CACHED file list
                hf_path = f"{orientation}/{unique_id}.mp4"
                
                if not uploader.file_exists(hf_path):
                    # 🛠️ DRAFT CAPTION
                    draft_desc = v.get("description", "").strip()
                    tag_str = ", ".join(tags[:5])
                    audio_note = " Includes original audio." if metadata.get("has_audio") else ""
                    
                    if draft_desc:
                        caption = f"{draft_desc}. Highlights: {tag_str}.{audio_note}"
                    else:
                        caption = f"A video featuring {tag_str}.{audio_note}"

                    video_metadata = {
                        "v_id": v_id,
                        "source": source_name,
                        "url": best_link,
                        "duration": v.get("duration"),
                        "orientation": orientation,
                        "tags": tags,
                        "description": draft_desc,
                        "caption": caption,
                        "technical": metadata
                    }
                    
                    print(f"  🎬 Saving {unique_id}...")
                    try:
                        uploader.upload_video(local_name, hf_path)
                        uploader.upload_metadata(video_metadata, hf_path.replace(".mp4", ".json"))
                        uploader.upload_text(caption, hf_path.replace(".mp4", ".txt"))
                        # Update the local cache so other threads don't try to upload it
                        uploader.existing_files.add(hf_path)
                    except Exception as e:
                        print(f"  Upload error: {e}")
            
            if os.path.exists(local_name):
                os.remove(local_name)

    # Use ThreadPool to process videos in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 1. INITIAL SEED SEARCH
        for query in QUERIES:
            print(f"\n--- Initial Seed: {query} ---")
            for client in clients:
                source_name = client.__class__.__name__
                try:
                    videos = client.search_videos(query, per_page=ITEMS_PER_QUERY)
                    for v in videos:
                        executor.submit(process_single_video, v, client, source_name, True)
                except Exception as e:
                    print(f"Failed to search {source_name}: {e}")

        # 2. DISCOVERY LOOP: Search for tags found during seeds
        # We wait a moment for seeds to populate collected_tags
        time.sleep(5)
        discovery_queries = list(collected_tags)[:10] # limit discovery depth
        print(f"\n🚀 DISCOVERY MODE: Found {len(collected_tags)} tags. Exploring: {discovery_queries}")
        
        for query in discovery_queries:
            for client in clients:
                source_name = client.__class__.__name__
                try:
                    videos = client.search_videos(query, per_page=5)
                    for v in videos:
                        executor.submit(process_single_video, v, client, source_name, False)
                except:
                    continue

if __name__ == "__main__":
    main()
