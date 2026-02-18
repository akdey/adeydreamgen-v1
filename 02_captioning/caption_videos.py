"""
02_captioning/caption_videos.py

PHASE 2: The Vision Brain
Uses BLIP-2 to generate high-quality captions for videos downloaded in Phase 1.
This script should be run separately from training to avoid VRAM OOM.
"""
import os
import json
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import cv2
import glob

# Configuration
CONFIG = {
    "video_dir": "../01_data_collection/videos",  # Adjust path as needed
    "output_json": "captions.json"
}

def setup_blip():
    print("🧠 Loading BLIP-2 Vision Model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return processor, model

def caption_video(video_path, processor, model):
    try:
        # Extract middle frame
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return None
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, text="a cinematic shot of", return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        if not caption.startswith("a cinematic shot of"):
            caption = "a cinematic shot of " + caption
            
        return caption
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def main():
    if not os.path.exists(CONFIG["video_dir"]):
        print(f"❌ Video directory not found: {CONFIG['video_dir']}")
        return

    processor, model = setup_blip()
    
    video_files = glob.glob(os.path.join(CONFIG["video_dir"], "*.mp4"))
    print(f"Found {len(video_files)} videos.")
    
    captions = {}
    for i, vid_path in enumerate(video_files):
        print(f"[{i+1}/{len(video_files)}] Captioning {os.path.basename(vid_path)}...")
        cap = caption_video(vid_path, processor, model)
        if cap:
            captions[os.path.basename(vid_path)] = cap
            print(f"   -> {cap}")
            
    with open(CONFIG["output_json"], "w") as f:
        json.dump(captions, f, indent=2)
    print(f"✅ Saved {len(captions)} captions to {CONFIG['output_json']}")

if __name__ == "__main__":
    main()
