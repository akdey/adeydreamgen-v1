# 🌌 ADeyDreamGen-v1 (Project: AMIT-VIDEO)

> **Building a High-Fidelity (8/10) Open-Source Video Generation Model.**

[![Scraping Automation](https://github.com/akdey/adeydreamgen-v1/actions/workflows/data_scrapper.yml/badge.svg)](https://github.com/akdey/adeydreamgen-v1/actions/workflows/data_scrapper.yml)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/a-k-dey/akd-video-training-dataset)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/models/a-k-dey/akd-video-v1)

**ADeyDreamGen-v1** is an ambitious project to push the boundaries of open-source video generation. By leveraging a custom-built, autonomous ingestion pipeline and the Zeroscope v2 XL spine, we are creating a model optimized for both **Cinematic Landscape (16:9)** and **Viral Portrait (9:16)** content.

---

## 🚀 The "WOW" Factors

### 🧠 1. Unbounded Recursive Discovery
Unlike static scrapers, our **Scout Engine** uses recursive tag-walking. It extracts metadata from high-quality videos and uses discovered concepts to "branch out" into the long-tail of visual data autonomously. It doesn't just scrape; it *explores*.

### 🎨 2. 3-Stage Aesthetic Filtering (Phase 1.5)
We don't train on noise. Our pipeline implements a rigorous quality gate:
1.  **Technical Filter**: Resolution (1080p+), Aspect Ratio verification, and Temporal Stability.
2.  **Semantic Mapping**: JSON-based metadata packaging for rich caption-to-video alignment.
3.  **Aesthetic Scoring**: Future integration of CLIP-based scoring to select only the top 1% of most "cinematic" frames for fine-tuning.

### 🏗️ 3. Temporal Mastery Fine-Tuning
Using a specialized training regime on **Kaggle's T4/P100 infrastructure**, we are fine-tuning the temporal layers of Zeroscope. This focuses specifically on smoothing high-motion scenes like drone shots, fluid dynamics, and human activities.

---

## 🛠️ Tech Stack & Infrastructure

- **Core Intelligence**: Zeroscope v2 XL (Spine)
- **Ingestion**: Multi-threaded Python Scraper (Pexels + Pixabay APIs)
- **Orchestration**: GitHub Actions (8-Way Parallel Matrix)
- **Storage**: Hugging Face Datasets (LFS for high-bitrate MP4s)
- **Compute**: Kaggle GPU (Phase 2 Training)

---

## 📈 Roadmap

- [x] **Phase 1: The Great Harvesting** – Modular scraper with dual-source ingestion.
- [x] **Phase 1.1: Parallelization** – Matrix-based GitHub automation (32x concurrent threads).
- [x] **Phase 1.2: Recursive Exploration** – Unbounded tag-based discovery.
- [ ] **Phase 2: The Vision Brain** – AI-driven hyper-descriptive captioning (LLaVA-1.5).
- [ ] **Phase 3: The Forge** – Fine-tuning on 1000+ curated cinematic clips.
- [ ] **Phase 4: The Generation** – Deployment of the ADeyDreamGen-v1 model.

---

## 📂 Project Structure

```text
├── .github/workflows/    # Parallel Matrix Automation
├── data_scrapper/        # The Scout Engine (Python)
│   ├── src/data_scraper/ # Core Logic (Clients, Processors, Multi-threading)
│   └── categories.py     # Global Class Definitions
├── training/             # Kaggle Fine-tuning Notebooks (Planned)
└── evaluation/           # Quality Metric Scripts (FVD, CLIP Score)
```

---

## 🤝 Stay Tuned
This project is under active development. My mission is to prove that high-quality generative video is achievable through smart engineering and open-source synergy.

*Designed & Engineered by [A.K. Dey](https://github.com/akdey)*
