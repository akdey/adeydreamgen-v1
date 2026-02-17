from huggingface_hub import HfApi
import os

class HFUploader:
    def __init__(self, repo_id: str = None, token: str = None):
        self.repo_id = repo_id or os.getenv("HF_REPO_ID")
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)
        
        if not self.repo_id:
            raise ValueError("HF_REPO_ID is required")

    def file_exists(self, hf_path: str) -> bool:
        """Check if a file exists in the Hugging Face repo."""
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            return hf_path in files
        except Exception:
            return False

    def upload_video(self, local_path: str, hf_path: str, commit_message: str = "Add new video"):
        """Upload a video file to the Hugging Face dataset repo."""
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=commit_message
        )

    def upload_metadata(self, metadata: dict, hf_path: str):
        """Upload metadata as a JSON file to Hugging Face."""
        import json
        import io
        
        content = json.dumps(metadata, indent=2).encode("utf-8")
        file_obj = io.BytesIO(content)
        
        self.api.upload_file(
            path_or_fileobj=file_obj,
            path_in_repo=hf_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add metadata for {hf_path}"
        )
