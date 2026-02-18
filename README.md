# 🎬 ADeyDreamGen-v1 (Open Source Video Model)

This project is a complete pipeline for training cinematic text-to-video models on consumer hardware (T4 GPU). 
The goal: **Democratize high-quality video generation.**

Most papers use A100 clusters. This repo is optimized for the rest of us. 
It includes a smart scraper, a vision-captioning bot, and a highly-optimized training script that fits 1.7B parameters into 15GB VRAM.

---

## 🧭 The Explorer Bot (`01_data_collection`)
I didn't want a static dataset. I wanted diversity.
The scraper isn't just a downloader—it's an **Explorer**.

1.  It starts with a seed (e.g., "cinematic mist").
2.  It downloads the best videos.
3.  **It reads the tags.** If a video is tagged with "witcher", the *next* run will search for "witcher".
4.  This creates an organic, ever-expanding dataset ("Forest" → "Mist" → "Dark Fantasy" → "Medieval"...).

**To run it:**
```bash
python 01_data_collection/scraper.py
```
It runs every 3 hours via GitHub Actions to keep the dataset fresh.

---

## 🧠 The Vision Brain (`02_captioning`)
Raw metadata sucks. "Forest, 4k" teaches the model nothing.
I use **BLIP-2** (a vision-language model) to watch the videos and write proper descriptions:
> *"A cinematic drone shot flying through a misty pine forest at sunrise, god rays piercing through the trees."*

Run this separately on a GPU to "upgrade" your dataset before training.

---

## 🏋️ The Training (`03_training`)
This is the hard part. Fitting Zeroscope XL on a T4 GPU.
I had to use every trick in the book:
- **8-bit AdamW Optimizer** (cuts VRAM usage in half)
- **CPU Offloading** (Text Encoder never touches GPU)
- **VAE Slicing** (Encodes video in tiny chunks)
- **16 Frames @ 256p** (The absolute limit for a T4)

It works. You can fine-tune a state-of-the-art video model for free on Kaggle.

**To train:**
1. Open `03_training/train_model.py` in Kaggle.
2. Add your HF_TOKEN.
3. Hit Run.

---

## 🔓 Unlock High Quality (Rich Mode)
If you have an A100 (40GB+ VRAM), you can train the **Real Deal** (576x320 Cinematic).
Just update `03_training/train_model.py`:
```python
CONFIG = {
    "resolution": 576,        # T4 can't handle this
    "num_frames": 24,         # Smoother motion
    "train_batch_size": 2,    
    "use_ai_captioning": True # Enable the Vision Brain
}
```

---

## 🚀 Deployment (`05_deployment`)
The final app is a Gradio interface. It runs on CPU or GPU and uses **LoRA weights** to inject your trained style into the base model.
Upload the folder to Hugging Face Spaces to share it with the world.

---

### Credits
Built with ❤️ and most of the help with ai tools, coffee, and a lot of OutOfMemory errors.
- **Base Model**: Zeroscope v2 XL
- **Tech Stack**: Diffusers, PyTorch, PEFT (LoRA)
