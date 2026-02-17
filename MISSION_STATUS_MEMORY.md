# 🎯 PROJECT: ADeyDreamGen-v1 (MISSION STATUS)

## 🏁 CURRENT OBJECTIVE
Building a High-Quality (8/10) Video Generation Model using 100% Free/Open-Source infrastructure, optimized for both YouTube Shorts (9:16) and Landscape (16:9).

---

## 🏗️ INFRASTRUCTURE SETUP
1.  **Command Center**: GitHub Repo (VideoGenModel-akd-V1)
2.  **Storage (Dataset)**: [a-k-dey/akd-video-training-dataset](https://huggingface.co/datasets/a-k-dey/akd-video-training-dataset) (Private)
3.  **Brain (Model)**: [akdey/adeydreamgen-v1](https://huggingface.co/models/akdey/adeydreamgen-v1) (Private)
4.  **Training Hub**: Kaggle (30 hrs/week free T4/P100 GPU)

---

## 🛠️ COMPLETED COMPONENTS
- ✅ **Base Architecture**: Selection of Zeroscope v2 XL as the backbone.
- ✅ **Smart Scraper v2**: Supported both Pexels and Pixabay with deduplication and 1080p+ quality filtering.
- ✅ **Automation**: `.github/workflows/data_scrapper.yml` set to run every 6 hours.
- ✅ **Verification**: `data_scrapper/src/scripts/verify_setup.py` created for API/Repo health checks.
- ✅ **Git Setup**: Repo initialized as `VideoGenModel-akd-V1` and pushed to `akdey` on GitHub.

---

## 📊 DATA STRATEGY (PHASE 1)
- **Sources**: Pexels, Pixabay, and YouTube (via `yt-dlp`).
- **Quality**: 1080p+, professionally shot, no watermarks.
- **Filtering**: 3-15 seconds duration, automatic Landscape/Portrait categorization.
- **Target**: 2,000 - 5,000 highly curated cinematic clips.

---

## 🧪 TRAINING STRATEGY (PHASE 2 - UPCOMING)
- **Method**: Fine-tuning temporal layers of Zeroscope/ModelScope.
- **Gym**: Kaggle GPU session.
- **Monitoring**: Weights & Biases (W&B) integration for loss curves and sample generation.
- **Quality Checks**: FVD (Fréchet Video Distance) and CLIP Score for prompt alignment.

---

## 🔑 REQUIRED SECRETS (GITHUB & .ENV)
- `PEXELS_API_KEY`: For video scraping.
- `PIXABAY_API_KEY`: For video scraping.
- `HF_TOKEN`: (Write access) for uploading to Dataset and Model repos.
- `HF_REPO_ID`: `a-k-dey/akd-video-training-dataset`

---

## 🚀 NEXT STEPS FOR NEW SESSION
1.  **Automate**: Finalize the GitHub Push and verify the first "Action" runs successfully.
2.  **Kaggle Setup**: Create the Python script for the Kaggle Training session once the dataset hits ~100 videos.
3.  **Captioning**: Implement a Vision-LLM step to generate high-quality captions for the training dataset.

---

## 🗒️ CHAT MEMORY (SESSION LOGS)
- **2026-02-18**: 
  - Segregated project into `data_scrapper`, `training`, and `evaluation` modules.
  - Renamed project to **ADeyDreamGen-v1**.
  - Fixed GitHub Actions issue: replaced obsolete `libgl1-mesa-glx` with `libgl1`.
  - Discussion on W&B (Weights & Biases) for training monitoring.
  - Discussion on 8/10 quality target via 3-stage filtering (Resolution -> Temporal -> AI Aesthetic Scoring).

