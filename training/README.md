# 🏗️ Training Module — ADeyDreamGen-v1

This module contains the complete pipeline for deploying, testing, fine-tuning, and iterating on the video generation model.

## 🚀 Quick Start (on Kaggle)

### Track A: See Baseline Performance (15 min)
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → New Notebook
2. **Settings** → Accelerator → **GPU T4 x2**
3. Copy `inference_zeroscope.py` into a cell
4. Run it → You'll get 5 sample videos showing "vanilla" Zeroscope quality

### Track B: Fine-Tune with Your Data (2-4 hours)
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → New Notebook
2. **Settings** → Accelerator → **GPU T4 x2**
3. **Add-ons** → Secrets → Add `HF_TOKEN` (your Hugging Face token)
4. Copy `finetune_zeroscope.py` into a cell
5. Run it → Model trains on your scraped dataset

---

## 📂 Files

| File | Purpose |
|------|---------|
| `inference_zeroscope.py` | **Track A**: Load pre-trained model, generate baseline videos |
| `finetune_zeroscope.py` | **Track B**: Fine-tune temporal layers using LoRA on your dataset |

---

## 🧠 Training Strategy: Temporal LoRA

We use **Low-Rank Adaptation (LoRA)** to fine-tune only the temporal (motion) layers:

```
┌─────────────────────────────────────┐
│        Zeroscope v2 XL              │
│  ┌──────────────┐  ┌─────────────┐  │
│  │ Spatial Layers│  │Temporal Layers│ │
│  │  (FROZEN ❄️)  │  │ (TRAINED 🔥) │ │
│  │ "What things  │  │ "How things  │ │
│  │  look like"   │  │   move"      │ │
│  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────┘
```

**Why LoRA?**
- Uses only **~8GB VRAM** (fits on Kaggle's free T4)
- Trains in **2-4 hours** instead of days
- Produces a tiny adapter file (~50MB) instead of a full model copy (~5GB)

---

## 📊 Monitoring

Training metrics are logged to **Weights & Biases** (if `use_wandb=True`):
- Loss curve (should decrease steadily)
- Learning rate schedule
- Checkpoint timestamps

---

## ⚙️ Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 24 | Frames per video (~3s at 8fps) |
| `resolution` | 320 | Training resolution (320=fast, 576=quality) |
| `lora_rank` | 16 | LoRA rank (higher=more capacity, more VRAM) |
| `learning_rate` | 1e-5 | Initial learning rate |
| `max_train_steps` | 2000 | Total training steps |
| `save_every_n_steps` | 500 | Checkpoint frequency |
