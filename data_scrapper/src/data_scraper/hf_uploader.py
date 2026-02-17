from huggingface_hub import HfApi
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

    def upload_video(self, local_path: str, hf_path: str, commit_message: str = "Add new video"):
        import time
        for attempt in range(3):
            try:
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=commit_message
                )
                return
            except Exception as e:
                print(f"⚠️ HF Upload failed (attempt {attempt+1}): {e}")
                time.sleep(10 * (attempt + 1))
        raise Exception("Failed to upload video to HF after 3 attempts")

    def upload_metadata(self, metadata: dict, hf_path: str):
        import json
        import io
        import time
        content = json.dumps(metadata, indent=2).encode("utf-8")
        for attempt in range(3):
            try:
                file_obj = io.BytesIO(content)
                self.api.upload_file(
                    path_or_fileobj=file_obj,
                    path_in_repo=hf_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Add metadata for {hf_path}"
                )
                return
            except Exception as e:
                print(f"⚠️ HF Metadata Upload failed: {e}")
                time.sleep(5 * (attempt + 1))

    def upload_text(self, text: str, hf_path: str):
        import io
        import time
        for attempt in range(3):
            try:
                file_obj = io.BytesIO(text.encode("utf-8"))
                self.api.upload_file(
                    path_or_fileobj=file_obj,
                    path_in_repo=hf_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Add text file for {hf_path}"
                )
                return
            except Exception as e:
                print(f"⚠️ HF Text Upload failed: {e}")
                time.sleep(5 * (attempt + 1))
