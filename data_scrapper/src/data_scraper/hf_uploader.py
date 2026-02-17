from huggingface_hub import HfApi, CommitOperationAdd
import os

class HFUploader:
    def __init__(self, repo_id: str = None, token: str = None):
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self.existing_files = set()
        
        if not self.repo_id:
            raise ValueError("HF_REPO_ID is required")

    def cache_repo_files(self):
        """Fetch all filenames from the repo once to avoid repeated API calls."""
        try:
            print(f"📊 Caching file list from {self.repo_id}...")
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            self.existing_files = set(files)
            print(f"✅ Cached {len(self.existing_files)} existing files.")
        except Exception as e:
            print(f"⚠️ Could not cache files: {e}")

    def file_exists(self, hf_path: str) -> bool:
        """Check if a file exists in the cached list or on Hugging Face."""
        return hf_path in self.existing_files

    def upload_bundle(self, local_video_path: str, video_metadata: dict, caption: str, hf_video_path: str):
        """Upload video, metadata, and caption in ONE single commit to avoid rate limits."""
        import json
        import io
        import time
        
        v_id = os.path.basename(hf_video_path).split('.')[0]
        json_content = json.dumps(video_metadata, indent=2).encode("utf-8")
        
        operations = [
            CommitOperationAdd(path_in_repo=hf_video_path, path_or_fileobj=local_video_path),
            CommitOperationAdd(path_in_repo=hf_video_path.replace(".mp4", ".json"), path_or_fileobj=io.BytesIO(json_content)),
            CommitOperationAdd(path_in_repo=hf_video_path.replace(".mp4", ".txt"), path_or_fileobj=io.BytesIO(caption.encode("utf-8")))
        ]
        
        for attempt in range(3):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    operations=operations,
                    commit_message=f"Add video bundle {v_id}",
                    repo_type="dataset"
                )
                return
            except Exception as e:
                print(f"⚠️ HF Bundle Upload failed (attempt {attempt+1}): {e}")
                if "128 per hour" in str(e):
                    print("🚨 CRITICAL: Commit limit hit. Throttling for 60s...")
                    time.sleep(60)
                else:
                    time.sleep(10 * (attempt + 1))
        raise Exception("Failed to upload bundle after 3 attempts")
