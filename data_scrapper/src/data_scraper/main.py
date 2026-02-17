
import os
import time
from pexels_client import PexelsClient
from video_processor import VideoProcessor
from hf_uploader import HFUploader
from dotenv import load_dotenv

load_dotenv()

def main():
    # Configuration
    QUERIES = ["cinematic", "nature", "drone", "travel", "cityscape"]
    ITEMS_PER_QUERY = 5
    HF_REPO_ID = os.getenv("HF_REPO_ID")
    
    client = PexelsClient()
    processor = VideoProcessor()
    uploader = HFUploader()
    
    os.makedirs("temp_videos", exist_ok=True)
    
    for query in QUERIES:
        print(f"Searching for: {query}")
        videos = client.search_videos(query, per_page=ITEMS_PER_QUERY)
        
        for v in videos:
            v_id = v['id']
            # Find a 1080p or higher video file
            best_file = None
            for f in v['video_files']:
                if f['width'] >= 1920 or f['height'] >= 1920:
                    best_file = f
                    break
            
            if not best_file:
                # Fallback to any file if 1080p not found, but we prefer 1080p
                best_file = v['video_files'][0]
            
            # Filter by duration (3-15s)
            duration = v.get('duration', 0)
            if not (3 <= duration <= 15):
                print(f"Skipping {v_id}: duration {duration}s not in range")
                continue
                
            local_name = f"temp_videos/{v_id}.mp4"
            print(f"Downloading {v_id}...")
            if processor.download_video(best_file['link'], local_name):
                metadata = processor.get_video_metadata(local_name)
                orientation = metadata.get('orientation', 'other')
                
                if orientation in ['landscape', 'portrait']:
                    hf_path = f"{orientation}/{v_id}.mp4"
                    
                    if uploader.file_exists(hf_path):
                        print(f"Skipping {v_id}: already exists in HF repo.")
                        if os.path.exists(local_name):
                            os.remove(local_name)
                        continue

                    print(f"Uploading {v_id} as {orientation}...")
                    try:
                        uploader.upload_video(local_name, hf_path, commit_message=f"Add {orientation} video {v_id}")
                    except Exception as e:
                        print(f"Upload failed for {v_id}: {e}")
                else:
                    print(f"Skipping {v_id}: orientation {orientation} not supported")
                
                # Cleanup
                if os.path.exists(local_name):
                    os.remove(local_name)
            
            # Simple rate limiting/breather
            time.sleep(2)

if __name__ == "__main__":
    main()
