"""
🔥 ADeyDreamGen-v1 — Fine-Tuning Script (Track B: Train & Improve)
=====================================================================
This script fine-tunes the Zeroscope v2 XL temporal layers using your
scraped dataset from Hugging Face. It is designed to run on Kaggle's
free T4/P100 GPU.

HOW TO RUN (Kaggle):
1. Create a new Kaggle notebook
2. Enable GPU (T4 x2 or P100)
3. Add your HF_TOKEN as a Kaggle Secret
4. Paste this entire script into a cell
5. Run the cell

STRATEGY:
- We freeze the spatial (image) layers — they already know "what things look like"
- We only train the temporal (motion) layers — teaching the model "how things move"
- This is called "Temporal LoRA" and is extremely efficient (uses <8GB VRAM)
"""

import torch
import os
import json
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================
# !pip install -q diffusers transformers accelerate peft datasets imageio[ffmpeg] safetensors huggingface_hub wandb opencv-python-headless

# ============================================================
# 2. CONFIGURATION
# ============================================================
CONFIG = {
    # Model
    "base_model": "cerspense/zeroscope_v2_XL",
    
    # Dataset
    "dataset_repo": "a-k-dey/akd-video-training-dataset",  # Your HF dataset
    "use_ai_captioning": False,                            # DISABLED: Skipping Phase 2 to fix loading error
    
    # Training
    "output_dir": "/kaggle/working/finetuned_model",
    "num_train_epochs": 3,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4, # Effective batch size = 4
    "learning_rate": 1e-5,
    "lr_scheduler": "cosine",
    "warmup_steps": 100,
    "max_train_steps": 2000,
    
    # LoRA
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["to_q", "to_v", "to_k", "to_out.0"],
    
    # Video Processing
    "resolution": 320,          # Reduced to avoid OOM
    "num_frames": 16,           # Reduced to 16 for T4 VRAM safety
    "fps": 8,
    
    # Monitoring
    "use_wandb": True,
    "wandb_project": "adeydreamgen-v1",
    
    # Checkpointing & Deployment
    "save_every_n_steps": 500,
    "push_to_hub": True,             # Automatically push best model to HF
    "hub_model_id": "a-k-dey/adeydreamgen-v1", # Your target model repo
    "validation_prompts": [
        "A person walking through a misty forest, cinematic",
        "Ocean waves at sunset, golden hour, aerial drone shot",
        "Close up of coffee being poured into a white cup, slow motion"
    ]
}

# ============================================================
# 3. DATASET LOADER
# ============================================================
def load_training_data(config):
    """
    Load video-caption pairs from Hugging Face dataset.
    Expects: orientation/video_id.mp4 + orientation/video_id.txt
    """
    from huggingface_hub import HfApi, hf_hub_download
    import cv2
    import numpy as np
    
    api = HfApi()
    print(f"📦 Loading dataset from {config['dataset_repo']}...")
    
    # List all files
    all_files = api.list_repo_files(repo_id=config["dataset_repo"], repo_type="dataset")
    
    # Find video files
    video_files = [f for f in all_files if f.endswith(".mp4")]
    print(f"   Found {len(video_files)} videos in dataset")
    
    # Build pairs
    pairs = []
    for vf in video_files:
        txt_file = vf.replace(".mp4", ".txt")
        if txt_file in all_files:
            pairs.append({"video": vf, "caption": txt_file})
    
    print(f"   Found {len(pairs)} video-caption pairs")
    return pairs

def extract_frames_from_video(video_path, num_frames, resolution):
    """Extract evenly-spaced frames from a video file."""
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None
    
    # Sample evenly spaced frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to target resolution (maintaining aspect ratio)
            h, w = frame.shape[:2]
            if w > h:  # Landscape
                new_w = int(resolution * 16 / 9)
                new_h = resolution
            else:  # Portrait
                new_w = resolution
                new_h = int(resolution * 16 / 9)
            frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return np.stack(frames)  # (num_frames, H, W, 3)


def caption_videos_with_blip(pairs, config):
    """
    PHASE 2: The Vision Brain
    Use a VLM (BLIP-2 or similar) to generate dense, descriptive captions
    for each video before training. This significantly improves prompt adherence.
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
    import cv2
    import numpy as np
    
    print("\n" + "=" * 40)
    print("🧠 PHASE 2: VISION BRAIN ACTIVATED")
    print("=" * 40)
    print("   Loading BLIP-2 for automated captioning...")
    
    # Load BLIP-2 (Lightweight optic-2.7b fits on T4)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    new_pairs = []
    
    print(f"   Captioning {len(pairs)} videos...")
    for i, pair in enumerate(pairs):
        try:
            # 1. Extract the middle frame
            cap = cv2.VideoCapture(pair["video"])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                continue
                
            # Convert to PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 2. Generate Caption
            inputs = processor(images=image, text="a cinematic shot of", return_tensors="pt").to("cuda", torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Clean up caption
            if not caption.startswith("a cinematic shot of"):
                caption = "a cinematic shot of " + caption
            
            # 3. Store new caption
            pair["caption_text"] = caption
            new_pairs.append(pair)
            
            if i % 5 == 0:
                print(f"   [{i}/{len(pairs)}] Generated: {caption}")
                
        except Exception as e:
            print(f"   ⚠️ Failed to caption {pair['video']}: {e}")
            # Fallback to original text file if BLIP fails
            with open(pair["caption"], "r") as f:
                pair["caption_text"] = f.read().strip()
            new_pairs.append(pair)
            
    # Free VRAM explicitly
    del model, processor        
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("🧠 Vision Brain unloaded. Memory cleared.")
    
    return new_pairs

class VideoDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for video-caption pairs."""
    
    def __init__(self, pairs, config):
        from huggingface_hub import hf_hub_download
        
        self.config = config
        self.cache_dir = "/kaggle/working/video_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Pre-download all videos
        print(f"⬇️ Downloading {len(pairs)} videos...")
        downloaded_pairs = []
        for i, p in enumerate(pairs):
            if len(downloaded_pairs) >= 2: break # 🧪 TEST MODE: Stop after 2 downloads
            try:
                # Download Video
                video_path = hf_hub_download(
                    repo_id=config["dataset_repo"],
                    filename=p["video"],
                    repo_type="dataset",
                    cache_dir=self.cache_dir
                )
                
                # Check if we already computed caption in Phase 2
                if "caption_text" in p:
                    caption_content = p["caption_text"]
                else:
                    # Fallback to downloading text file
                    caption_path = hf_hub_download(
                        repo_id=config["dataset_repo"],
                        filename=p["caption"],
                        repo_type="dataset",
                        cache_dir=self.cache_dir
                    )
                    with open(caption_path, "r") as f:
                        caption_content = f.read().strip()
                
                downloaded_pairs.append({
                    "video": video_path, 
                    "caption": caption_content
                })
            except Exception as e:
                print(f"   ⚠️ Skipping {p['video']}: {e}")
        
        # 🔥 Run Phase 2 Captioning if enabled
        if config.get("use_ai_captioning", True):
            self.local_pairs = caption_videos_with_blip(downloaded_pairs[:2], config) # 🧪 TEST MODE: Only 2 items
        else:
            self.local_pairs = downloaded_pairs[:2] # 🧪 TEST MODE: Only 2 items
            
        print(f"✅ Prepared {len(self.local_pairs)} pairs for training (TEST MODE)")
    
    def __len__(self):
        return len(self.local_pairs)
    
    def __getitem__(self, idx):
        pair = self.local_pairs[idx]
        
        # Use valid caption string directly
        caption = pair["caption"]
        
        # Extract frames
        frames = extract_frames_from_video(
            pair["video"],
            self.config["num_frames"],
            self.config["resolution"]
        )
        
        if frames is None:
            frames = torch.zeros(self.config["num_frames"], 3, self.config["resolution"], self.config["resolution"])
            caption = "error"
        else:
            frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        
        return {"pixel_values": frames, "text": caption}


# ============================================================
# 4. TRAINING LOOP
# ============================================================
def setup_training(config):
    """Setup model, optimizer, and training infrastructure."""
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from peft import LoraConfig, get_peft_model
    
    print("🔄 Loading base model for fine-tuning...")
    pipe = DiffusionPipeline.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,
    )
    
    # Extract the UNet
    unet = pipe.unet
    
    # 📉 VRAM OPTIMIZATION: XFormers (Critical for T4)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers efficient attention enabled")
    except Exception as e:
        print(f"⚠️ xformers not available: {e}")
        # Fallback: Slicing (slower but saves memory)
        pipe.enable_attention_slicing()
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=0.05,
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Move to GPU
    unet = unet.to("cuda", dtype=torch.float16)
    
    # Freeze everything except LoRA
    for name, param in unet.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    
    # Optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config["learning_rate"], weight_decay=0.01)
        print("✅ Using 8-bit AdamW optimizer (VRAM SAVER)")
    except ImportError:
        print("⚠️ BitsAndBytes not found. Using standard AdamW.")
        optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=0.01)
    
    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_train_steps"])
    
    print(f"✅ Training setup complete")
    print(f"   Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"   LoRA Rank: {config['lora_rank']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    
    return pipe, unet, optimizer, scheduler

def train(config):
    """Main training loop."""
    
    # Optional: Weights & Biases
    if config["use_wandb"]:
        try:
            import wandb
            # Try to get API key from environment or Kaggle secrets
            wandb_key = os.getenv("WANDB_API_KEY")
            if not wandb_key:
                try:
                    from kaggle_secrets import UserSecretsClient
                    user_secrets = UserSecretsClient()
                    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
                except:
                    pass
            
            if wandb_key:
                # 🔑 FORCE ENV VAR: This handles both 40-char Personal Keys and 80-char Service Keys
                os.environ["WANDB_API_KEY"] = wandb_key
                wandb.login(key=wandb_key)
                wandb.init(project=config["wandb_project"], config=config)
                print("✅ Weights & Biases initialized")
            else:
                print("⚠️ WANDB_API_KEY not found. Running offline.")
                config["use_wandb"] = False
        except Exception as e:
            print(f"⚠️ W&B Error: {e}, continuing without monitoring")
            config["use_wandb"] = False
    
    # Load data
    pairs = load_training_data(config)
    if len(pairs) == 0:
        print("❌ No training data found! Make sure the scraper has collected videos.")
        return
    
    dataset = VideoDataset(pairs, config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup model
    pipe, unet, optimizer, scheduler = setup_training(config)
    
    # Training
    print("\n" + "=" * 60)
    print("🔥 TRAINING STARTED")
    print("=" * 60)
    
    os.makedirs(config["output_dir"], exist_ok=True)
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(config["num_train_epochs"]):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= config["max_train_steps"]:
                break
            
            pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
            
            # Encode text
            from transformers import CLIPTokenizer
            tokenizer = pipe.tokenizer
            text_inputs = tokenizer(
                batch["text"],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            
            # 📉 VRAM HACK: CPU OFFLOAD
            # Move Text Encoder to CPU, run inference, then move result to GPU
            pipe.text_encoder.to("cpu")
            text_encoder = pipe.text_encoder
            text_inputs = tokenizer(
                batch["text"],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ) # Keep on CPU
            
            with torch.no_grad():
                # Run on CPU
                print("   Running Text Encoder on CPU...", end="\r")
                text_embeddings = text_encoder(text_inputs.input_ids)[0]
                text_embeddings = text_embeddings.to("cuda", dtype=torch.float16) # Move ONLY result to GPU
            
            # 📉 VRAM HACK: CPU OFFLOAD VAE
            # Process frames through VAE
            # FIX: Unpack shape properly before using b, f, c, h, w
            b, f, c, h, w = pixel_values.shape
            
            vae = pipe.vae.to("cuda", dtype=torch.float16)
            pixel_values_flat = pixel_values.reshape(b * f, c, h, w)
            
            # Encode in chunks to save VAE memory
            latents_list = []
            chunk_size = 4 # Process 4 frames at a time
            for i in range(0, pixel_values_flat.shape[0], chunk_size):
                chunk = pixel_values_flat[i:i+chunk_size]
                with torch.no_grad():
                    chunk_latents = vae.encode(chunk).latent_dist.sample()
                latents_list.append(chunk_latents)
            
            latents = torch.cat(latents_list, dim=0)
            latents = latents * vae.config.scaling_factor
            latents = latents.reshape(b, f, *latents.shape[1:])
            latents = latents.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
            
            # Move VAE back to CPU to save memory for UNet? 
            # Ideally yes, but let's see if this is enough.
            del vae
            torch.cuda.empty_cache()
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (b,), device="cuda")
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()
            
            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Logging
            if global_step % 10 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                lr = scheduler.get_last_lr()[0]
                print(f"   Step {global_step}/{config['max_train_steps']} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                
                if config["use_wandb"]:
                    import wandb
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": lr,
                        "epoch": epoch,
                        "step": global_step
                    })
            
            # Checkpoint
            if global_step > 0 and global_step % config["save_every_n_steps"] == 0:
                save_checkpoint(unet, config, global_step)
                
                # Track best model
                avg_loss = epoch_loss / max(num_batches, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(unet, config, "best")
            
            # Free VRAM
            del pixel_values, noise, noisy_latents, noise_pred, latents
            torch.cuda.empty_cache()
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n📊 Epoch {epoch+1}/{config['num_train_epochs']} complete | Avg Loss: {avg_epoch_loss:.4f}")
    
    # Final save
    save_checkpoint(unet, config, "final")
    
    print("\n" + "=" * 60)
    print("🏁 TRAINING COMPLETE")
    print(f"   Final Loss: {avg_epoch_loss:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model saved to: {config['output_dir']}")
    print("=" * 60)
    
    if config["use_wandb"]:
        import wandb
        wandb.finish()

def save_checkpoint(unet, config, step_label):
    """Save a LoRA checkpoint."""
    checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint-{step_label}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    unet.save_pretrained(checkpoint_dir)
    print(f"💾 Checkpoint saved: {checkpoint_dir}")


# ============================================================
# 5. GENERATE WITH FINE-TUNED MODEL (Validation)
# ============================================================
def generate_comparison(config, checkpoint_path="final"):
    """
    Generate videos with both the base model and your fine-tuned model
    side by side for comparison.
    """
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    from peft import PeftModel
    
    compare_dir = os.path.join(config["output_dir"], "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    for i, prompt in enumerate(config["validation_prompts"]):
        print(f"\n🎬 Generating comparison for: '{prompt[:50]}...'")
        
        # Generate with BASE model
        base_result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted",
            num_frames=24, height=320, width=576,
            num_inference_steps=40, guidance_scale=12.5
        )
        base_path = os.path.join(compare_dir, f"base_{i:03d}.mp4")
        export_to_video(base_result.frames[0], base_path, fps=8)
        
        # Load fine-tuned LoRA weights
        lora_path = os.path.join(config["output_dir"], f"checkpoint-{checkpoint_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        
        # Generate with FINE-TUNED model
        ft_result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted",
            num_frames=24, height=320, width=576,
            num_inference_steps=40, guidance_scale=12.5
        )
        ft_path = os.path.join(compare_dir, f"finetuned_{i:03d}.mp4")
        export_to_video(ft_result.frames[0], ft_path, fps=8)
        
        print(f"   Base: {base_path}")
        print(f"   Fine-tuned: {ft_path}")
        
        torch.cuda.empty_cache()
    
    print(f"\n✅ Comparison videos saved to {compare_dir}")


# ============================================================
# 6. PUSH TO HUGGING FACE
# ============================================================
def push_to_hub(config, checkpoint_path="best"):
    """Push the fine-tuned LoRA weights to Hugging Face."""
    from huggingface_hub import HfApi
    
    api = HfApi()
    lora_dir = os.path.join(config["output_dir"], f"checkpoint-{checkpoint_path}")
    
    target_repo = config["hub_model_id"]
    
    print(f"🚀 Pushing LoRA weights to {target_repo}...")
    api.upload_folder(
        folder_path=lora_dir,
        repo_id=target_repo,
        repo_type="model",
        commit_message=f"Upload ADeyDreamGen-v1 LoRA ({checkpoint_path})"
    )
    print(f"✅ Model pushed to https://huggingface.co/{target_repo}")


# ============================================================
# 7. MAIN (Refactored for Memory Safety)
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["all", "caption", "train"], help="Run 'caption' first, then 'train' in separate internal calls")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🔥 ADeyDreamGen-v1 — Mode: {args.mode.upper()}")
    print("=" * 60)
    
    if args.mode == "caption":
        # ONLY Download & Caption
        print("🧠 Running Phase 2: Captioning Only...")
        pairs = load_training_data(CONFIG)
        
        # Download & Caption
        dataset = VideoDataset(pairs, CONFIG) # This triggers download & captioning
        print("✅ Captioning complete. Metadata saved in cache.")
        
    elif args.mode == "train":
        # ONLY Train (Assume data is ready)
        print("🏋️ Running Phase 3: Training Only...")
        # Disable captioning for this run since it's already done
        CONFIG["use_ai_captioning"] = False 
        train(CONFIG)
        
        # Step 2: Comparison
        generate_comparison(CONFIG, checkpoint_path="best")
        
        # Step 3: Push
        if CONFIG.get("push_to_hub", False):
            push_to_hub(CONFIG, checkpoint_path="best")
            
    else:
        # Default: Try to run everything (May OOM)
        train(CONFIG)
        generate_comparison(CONFIG, checkpoint_path="best")
        if CONFIG.get("push_to_hub", False):
            push_to_hub(CONFIG, checkpoint_path="best")

if __name__ == "__main__":
    main()
