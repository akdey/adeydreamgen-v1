"""
🎬 ADeyDreamGen-v1 — Inference Script (Track A: Deploy & Test)
================================================================
This script loads the pre-trained Zeroscope v2 XL model and generates
sample videos from text prompts. Use it to establish a BASELINE before
fine-tuning.

HOW TO RUN (Kaggle):
1. Create a new Kaggle notebook
2. Enable GPU (T4 x2 recommended)
3. Paste this entire script into a cell
4. Run the cell

OUTPUT:
- Generated .mp4 files saved to /kaggle/working/outputs/
- A baseline_report.json with generation metadata
"""

import torch
import os
import json
import time
from datetime import datetime

# ============================================================
# 1. INSTALL DEPENDENCIES (Kaggle-compatible)
# ============================================================
# !pip install -q diffusers transformers accelerate imageio[ffmpeg] safetensors

# ============================================================
# 2. CONFIGURATION
# ============================================================
MODEL_ID = "cerspense/zeroscope_v2_XL"
OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test prompts covering different categories
TEST_PROMPTS = [
    # Human Activities
    "A woman dancing gracefully in a sunlit studio, cinematic lighting, high quality",
    # Nature
    "Aerial drone shot of ocean waves crashing on rocky cliffs at golden hour, 4k cinematic",
    # Urban/Tech
    "Neon-lit cyberpunk city street at night with rain reflections, cinematic slow motion",
    # Objects/Textures
    "Close up of colorful ink drops swirling in water, macro photography, stunning detail",
    # Abstract/Emotion
    "Time-lapse of a flower blooming in morning light, soft bokeh background, peaceful"
]

# Generation parameters
GEN_CONFIG = {
    "num_frames": 24,           # ~3 seconds at 8fps
    "height": 320,              # Use 320 for speed, 576 for quality
    "width": 576,               # Use 576 for speed, 1024 for quality
    "num_inference_steps": 40,  # Higher = better quality, slower
    "guidance_scale": 12.5,     # How closely to follow the prompt
}

# ============================================================
# 3. LOAD MODEL
# ============================================================
def load_model():
    """Load the Zeroscope v2 XL pipeline."""
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    
    print("🔄 Loading Zeroscope v2 XL...")
    start = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    
    # Use DPM++ solver for faster, higher quality generation
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimizations for Kaggle's T4 (16GB VRAM)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    return pipe, load_time

# ============================================================
# 4. GENERATE VIDEOS
# ============================================================
def generate_video(pipe, prompt, index):
    """Generate a single video from a text prompt."""
    print(f"\n🎬 [{index+1}/{len(TEST_PROMPTS)}] Generating: '{prompt[:60]}...'")
    
    start = time.time()
    
    result = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted, ugly, noisy, watermark, text",
        num_frames=GEN_CONFIG["num_frames"],
        height=GEN_CONFIG["height"],
        width=GEN_CONFIG["width"],
        num_inference_steps=GEN_CONFIG["num_inference_steps"],
        guidance_scale=GEN_CONFIG["guidance_scale"],
    )
    
    gen_time = time.time() - start
    
    # Export to MP4
    from diffusers.utils import export_to_video
    output_path = os.path.join(OUTPUT_DIR, f"baseline_{index:03d}.mp4")
    export_to_video(result.frames[0], output_path, fps=8)
    
    print(f"   ✅ Saved to {output_path} ({gen_time:.1f}s)")
    
    return {
        "prompt": prompt,
        "output_path": output_path,
        "generation_time_s": round(gen_time, 2),
        "config": GEN_CONFIG
    }

# ============================================================
# 5. BASELINE REPORT
# ============================================================
def save_report(results, load_time):
    """Save a JSON report of all baseline generations."""
    report = {
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "model_load_time_s": round(load_time, 2),
        "generation_config": GEN_CONFIG,
        "total_videos": len(results),
        "avg_generation_time_s": round(sum(r["generation_time_s"] for r in results) / len(results), 2),
        "results": results
    }
    
    report_path = os.path.join(OUTPUT_DIR, "baseline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Baseline report saved to {report_path}")
    print(f"   Model: {MODEL_ID}")
    print(f"   Total Videos: {len(results)}")
    print(f"   Avg Generation Time: {report['avg_generation_time_s']}s per video")
    
    return report

# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("🌌 ADeyDreamGen-v1 — Baseline Inference")
    print("=" * 60)
    
    pipe, load_time = load_model()
    
    results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        result = generate_video(pipe, prompt, i)
        results.append(result)
        
        # Free up VRAM between generations
        torch.cuda.empty_cache()
    
    report = save_report(results, load_time)
    
    print("\n" + "=" * 60)
    print("🏁 BASELINE COMPLETE")
    print("   Now you can see the 'vanilla' Zeroscope output.")
    print("   Next step: Fine-tune with YOUR dataset to beat this baseline!")
    print("=" * 60)

if __name__ == "__main__":
    main()
