# 🎯 PROJECT: AMIT-VIDEO-V1 (MISSION STATUS)

## 🏁 CURRENT OBJECTIVE
Building a High-Quality (8/10) Video Generation Model using 100% Free/Open-Source infrastructure, optimized for both YouTube Shorts (9:16) and Landscape (16:9).

---

## 🏗️ INFRASTRUCTURE SETUP
1.  **Command Center**: GitHub Repo (VideoGenModel-akd-V1)
2.  **Storage (Dataset)**: [a-k-dey/akd-video-training-dataset](https://huggingface.co/datasets/a-k-dey/akd-video-training-dataset) (Private)
3.  **Brain (Model)**: [a-k-dey/akd-video-v1](https://huggingface.co/models/a-k-dey/akd-video-v1) (Private)
4.  **Training Hub**: Kaggle (30 hrs/week free T4/P100 GPU)

---

## 🛠️ COMPLETED COMPONENTS
- ✅ **Base Architecture**: Selection of Zeroscope v2 XL as the backbone.
- ✅ **Smart Scraper**: `data_scrapper/src/scripts/smart_scraper.py` configured to detect aspect ratios (Portrait/Landscape) and upload directly to HF.
- ✅ **Automation**: `.github/workflows/data_scrapper.yml` set to run every 6 hours.
- ✅ **Verification**: `data_scrapper/src/scripts/verify_setup.py` created for API/Repo health checks.

---

## 📊 DATA STRATEGY (PHASE 1)
- **Source**: Pexels/Pixabay (Royalty-free).
- **Quality**: 1080p+, No Watermarks, 3-15 seconds duration.
- **Bucketing**: Multi-aspect ratio support (Landscape vs Portrait) in a single unified model.
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
- `HF_TOKEN`: (Write access) for uploading to Dataset and Model repos.
- `HF_REPO_ID`: `a-k-dey/akd-video-training-dataset`

---

## 🚀 NEXT STEPS FOR NEW SESSION
1.  **Automate**: Finalize the GitHub Push and verify the first "Action" runs successfully.
2.  **Kaggle Setup**: Create the Python script for the Kaggle Training session once the dataset hits ~100 videos.
3.  **Captioning**: Implement a Vision-LLM step to generate high-quality captions for the training dataset.

---
*Created on 2026-02-18 | Session: Antigravity-Video-Factory-v1*
