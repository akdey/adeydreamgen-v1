# Data Scraper Module

This module handles the automated collection of high-quality, royalty-free video data for the VideoGen project.

## Features
- **Multi-Source Integration**: 
  - **Pexels API**: Professional royalty-free clips.
  - **Pixabay API**: High-quality film/cinematic clips.
  - **YouTube (Curated)**: *Upcoming* support for high-quality drone/nature channels using `yt-dlp`.
- **Smart Filtering**: 
  - Resolution: 1080p+ preferred.
  - Duration: 3-15 seconds.
  - Aspect Ratio: Categorizes into `landscape` (16:9) and `portrait` (9:16).
- **Automated Upload**: Directly uploads to Hugging Face Datasets.
- **Deduplication**: Checks current repository content to avoid redundant uploads.

## Structure
- `pexels_client.py`: API wrapper for Pexels.
- `video_processor.py`: Uses OpenCV to verify video properties.
- `hf_uploader.py`: Handles Hugging Face repository interactions.
- `main.py`: Orchestrator script.

## Environment Variables
Required in `.env` or GitHub Secrets:
- `PEXELS_API_KEY`
- `HF_TOKEN`
- `HF_REPO_ID`
