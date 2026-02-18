import gradio as gr
import torch
import os
import random
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "cerspense/zeroscope_v2_XL"
LORA_MODEL_ID = "a-k-dey/adeydreamgen-v1" # Your Hugging Face trained model
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Model (Cached) ---
def load_pipeline():
    print("🔄 Loading Base Model...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Optimize for T4 GPU (Hugging Face Spaces)
    # CPU Offload: Moves unused model parts to CPU RAM (Critical for 16GB VRAM)
    pipe.enable_model_cpu_offload() 
    
    # VAE Slicing/Tiling: Process images in chunks to save VRAM
    pipe.enable_vae_slicing()
    # pipe.enable_vae_tiling() # Enable if you see black images at high res
    
    print(f"🔄 Loading LoRA Weights from {LORA_MODEL_ID}...")
    try:
        # Check if LoRA exists on Hub before loading
        from huggingface_hub import list_repo_files
        files = list_repo_files(LORA_MODEL_ID)
        if "adapter_config.json" in files:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_MODEL_ID)
            print("✅ LoRA weights loaded successfully!")
        else:
            print("⚠️ LoRA not found on Hub yet. Running Base Model.")
    except Exception as e:
        print(f"⚠️ Could not load LoRA (using base model only): {e}")

    return pipe

# Initialize pipeline globally
pipeline = load_pipeline()

def generate(prompt, negative_prompt, steps, guidance, seed):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    print(f"🎬 Generating: {prompt}")
    start_time = time.time()
    
    video_frames = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=576,
        height=320,
        num_frames=24,
        generator=generator
    ).frames[0]
    
    output_path = f"{OUTPUT_DIR}/generated_{int(time.time())}.mp4"
    export_to_video(video_frames, output_path, fps=8)
    
    gen_time = time.time() - start_time
    print(f"✅ Video saved in {gen_time:.1f}s")
    
    return output_path, f"Seed: {seed} | Time: {gen_time:.1f}s"

# --- Gradio UI ---
with gr.Blocks(title="ADeyDreamGen-v1 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌌 ADeyDreamGen-v1
        **A Cinematic Video Generation Model** by A.K. Dey
        
        Generate 3-second cinematic clips (576x320) using our custom fine-tuned model.
        """
    )
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="A cinematic drone shot of a futuristic city at sunset...")
            neg_prompt_input = gr.Textbox(label="Negative Prompt", value="low quality, blurry, distorted, watermark, text, ugly")
            
            with gr.Accordion("Advanced Settings", open=False):
                steps_slider = gr.Slider(minimum=10, maximum=100, value=40, step=1, label="Inference Steps")
                guidance_slider = gr.Slider(minimum=1, maximum=20, value=12.5, step=0.1, label="Guidance Scale")
                seed_input = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
            
            generate_btn = gr.Button("🚀 Generate Video", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            info_output = gr.Textbox(label="Info", interactive=False)
            
    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, neg_prompt_input, steps_slider, guidance_slider, seed_input],
        outputs=[video_output, info_output]
    )
    
    gr.Examples(
        examples=[
            ["A woman walking in a misty forest, cinematic lighting", "low quality, blurry", 40, 12.5, -1],
            ["Drone shot of ocean waves crashing on rocks, golden hour", "low quality, blurry", 40, 12.5, -1],
            ["Cyberpunk street with neon signs, rain reflections", "low quality, blurry", 40, 12.5, -1],
        ],
        inputs=[prompt_input, neg_prompt_input, steps_slider, guidance_slider, seed_input]
    )

if __name__ == "__main__":
    demo.queue().launch()
