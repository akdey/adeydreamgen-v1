"""
04_evaluation/generate_samples.py

PHASE 4: Evaluation
Loads the fine-tuned LoRA model and generates standardized test samples 
to compare progress visually.
"""
import torch
import os
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "cerspense/zeroscope_v2_XL"
LORA_MODEL_ID = "../03_training/finetuned_model" # Local path or HF ID
OUTPUT_DIR = "samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test Prompts
PROMPTS = [
    "A drone shot of ocean waves crashing on rocks, golden hour, cinematic lighting",
    "A woman walking in a misty forest, cinematic lighting, 4k",
    "Cyberpunk street with neon signs, rain reflections, futuristic city",
    "Time lapse of clouds moving over a mountain range, dramatic sky"
]

def load_pipeline():
    print("🔄 Loading Base Model...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # optimize for T4
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    print(f"🔄 Loading LoRA from {LORA_MODEL_ID}...")
    try:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_MODEL_ID)
        print("✅ LoRA weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load LoRA (using base model only): {e}")

    return pipe

def main():
    pipe = load_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(42)
    
    print(f"🎬 Generating {len(PROMPTS)} test samples...")
    
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt}")
        frames = pipe(
            prompt, 
            negative_prompt="low quality, blurry, distorted, watermark, text, ugly",
            num_frames=16, 
            width=576, 
            height=320, 
            num_inference_steps=30,
            guidance_scale=12.5,
            generator=generator
        ).frames[0]
        
        # Save
        filename = f"sample_{i+1}_{prompt[:20].replace(' ', '_')}.mp4"
        path = os.path.join(OUTPUT_DIR, filename)
        export_to_video(frames, path, fps=8)
        print(f"   Saved to {path}")

if __name__ == "__main__":
    main()
