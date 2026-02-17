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

def main():
    parser = argparse.ArgumentParser(description="ADeyDreamGen-v1 Data Scrapper")
    parser.add_argument("--query", type=str, help="Search query (e.g. cinematic, drone)")
    parser.add_argument("--items", type=int, default=15, help="Number of items to fetch per source")
    args = parser.parse_args()

    # Configuration
    QUERIES = [args.query] if args.query else ["cinematic", "nature", "drone", "travel", "cityscape", "ocean", "mountain", "forest"]
    ITEMS_PER_QUERY = args.items
    
    # Initialize components
    processor = VideoProcessor()
    uploader = HFUploader()
    
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
    
    def process_single_video(v, client, source_name):
        v_id = v['id']
        # Link extraction
        if isinstance(client, PexelsClient):
            best_file = None
            for f in v.get('video_files', []):
                if f['width'] >= 1920 or f['height'] >= 1920:
                    best_file = f
                    break
            best_link = best_file['link'] if best_file else v.get('video_files', [{}])[0].get('link')
        else: # Pixabay
            best_link = client.get_best_video_link(v)
        
        if not best_link:
            return

        # Filter by duration (3-15s)
        duration = v.get('duration', 0)
        if not (3 <= duration <= 15):
            print(f"  Skipping {v_id}: duration {duration}s not in range")
            return
            
        local_name = f"temp_videos/{v_id}_{source_name}.mp4"
        
        print(f"  Downloading {v_id} from {source_name}...")
        if processor.download_video(best_link, local_name):
            metadata = processor.get_video_metadata(local_name)
            orientation = metadata.get('orientation', 'other')
            
            if orientation in ['landscape', 'portrait']:
                hf_path = f"{orientation}/{v_id}.mp4"
                
                if uploader.file_exists(hf_path):
                    print(f"  Skipping {v_id}: already exists in HF repo.")
                else:
                    print(f"  Uploading {v_id} as {orientation}...")
                    try:
                        uploader.upload_video(local_name, hf_path, commit_message=f"Add {orientation} video {v_id} from {source_name}")
                    except Exception as e:
                        print(f"  Upload failed for {v_id}: {e}")
            else:
                print(f"  Skipping {v_id}: orientation {orientation} not supported")
            
            # Cleanup
            if os.path.exists(local_name):
                os.remove(local_name)

    # Use ThreadPool to process videos in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for query in QUERIES:
            print(f"\n--- Searching for: {query} ---")
            for client in clients:
                source_name = client.__class__.__name__
                try:
                    videos = client.search_videos(query, per_page=ITEMS_PER_QUERY)
                    for v in videos:
                        executor.submit(process_single_video, v, client, source_name)
                except Exception as e:
                    print(f"Failed to search {source_name}: {e}")
                    continue

if __name__ == "__main__":
    main()
