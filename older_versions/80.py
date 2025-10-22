import gradio as gr
import sys
import subprocess
import os
import platform
import shutil
import time
import threading
import queue
import argparse
import copy
import json

try:
    from huggingface_hub import hf_hub_download, snapshot_download, HfFileSystem
    from huggingface_hub.utils import HfHubHTTPError, HFValidationError
except ImportError:
    print("huggingface_hub not found. Attempting installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.20.0"]) # Added version specifier
        import importlib
        importlib.invalidate_caches()
        from huggingface_hub import hf_hub_download, snapshot_download, HfFileSystem
        from huggingface_hub.utils import HfHubHTTPError, HFValidationError
        print("huggingface_hub installed and imported successfully.")
    except Exception as e:
        print(f"ERROR: Failed to install or import huggingface_hub: {e}")
        print("Please install it manually: pip install huggingface_hub>=0.20.0")
        sys.exit(1)


APP_TITLE = f"Unified AI Models Downloader for SwarmUI, ComfyUI, Automatic1111 and Forge Web UI"

def install_package(package_name, version_spec=""):
    """Installs a package using pip."""
    try:
        print(f"Attempting to install {package_name}{version_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}{version_spec}"])
        print(f"Successfully installed {package_name}.")
        if package_name == "huggingface_hub":
            import importlib
            importlib.invalidate_caches()
            globals()['hf_hub_download'] = importlib.import_module('huggingface_hub').hf_hub_download
            globals()['snapshot_download'] = importlib.import_module('huggingface_hub').snapshot_download
            globals()['HfFileSystem'] = importlib.import_module('huggingface_hub').HfFileSystem
            globals()['HfHubHTTPError'] = importlib.import_module('huggingface_hub.utils').HfHubHTTPError
            globals()['HFValidationError'] = importlib.import_module('huggingface_hub.utils').HFValidationError
        elif package_name == "hf_transfer":
             import importlib
             importlib.invalidate_caches()
             globals()['HF_TRANSFER_AVAILABLE'] = True
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install {package_name}: {e}")
        print("Please install it manually using: pip install ", f"{package_name}{version_spec}")
    except ImportError:
         print(f"ERROR: Failed to import {package_name} even after attempting install.")
    return False

print("huggingface_hub found (or installed).")

try:
    import hf_transfer
    print("hf_transfer found.")
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    print("hf_transfer is optional but recommended for faster downloads.")
    HF_TRANSFER_AVAILABLE = False
    if install_package("hf_transfer", ">=0.1.8"):
        try:
            import hf_transfer
            print("hf_transfer installed successfully after attempt.")
            HF_TRANSFER_AVAILABLE = True
        except ImportError:
            print("hf_transfer still not found after install attempt.")
            HF_TRANSFER_AVAILABLE = False
    else:
        HF_TRANSFER_AVAILABLE = False

HIDREAM_INFO_LINK = "https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#hidream-i1"
GGUF_QUALITY_INFO = "GGUF Quality: Q8 > Q6 > Q5 (K_M > K_S > 1 > 0) > Q4 (K_M > K_S > 1 > 0) > Q3 (K_M > K_S) > Q2_K."

# Define the new VAE model entry here to be referenced in models_structure
ltx_vae_companion_entry = {
    "name": "LTX VAE (BF16) - Companion for LTX 13B Dev Models",
    "repo_id": "wsbagnsv1/ltxv-13b-0.9.7-dev-GGUF",
    "filename_in_repo": "ltxv-13b-0.9.7-vae-BF16.safetensors",
    "save_filename": "LTX_VAE_13B_Dev_BF16.safetensors",
    "target_dir_key": "vae"  # Ensures it's saved in the VAE folder
}

wan_causvid_14b_lora_v2_entry = {
    "name": "Wan 2.1 CausVid T2V/I2V LoRA v2 14B (Rank 32) - Companion",
    "repo_id": "Kijai/WanVideo_comfy",
    "filename_in_repo": "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
    "save_filename": "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
    "target_dir_key": "Lora",
    "info": "High-speed LoRA v2 for Wan 2.1 14B T2V/I2V. Saves to Lora folder. Also listed under 'Wan 2.1 Models' and 'LoRA Models'. See SwarmUI Video Docs for usage details on CFG, Steps, FPS, and Trim Video Start Frames."
}

wan_causvid_14b_lora_entry = {
    "name": "Wan 2.1 CausVid T2V/I2V LoRA 14B (Rank 32) - Companion",
    "repo_id": "Kijai/WanVideo_comfy",
    "filename_in_repo": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    "save_filename": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    "target_dir_key": "Lora",
    "info": "High-speed LoRA for Wan 2.1 14B T2V/I2V. Saves to Lora folder. Also listed under 'Wan 2.1 Models' and 'LoRA Models'. See SwarmUI Video Docs for usage details on CFG, Steps, FPS, and Trim Video Start Frames."
}

wan_causvid_1_3b_lora_entry = {
    "name": "Wan 2.1 CausVid T2V LoRA 1.3B (Rank 32) - Companion",
    "repo_id": "Kijai/WanVideo_comfy",
    "filename_in_repo": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    "save_filename": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    "target_dir_key": "Lora",
    "info": "High-speed LoRA for Wan 2.1 1.3B T2V. Saves to Lora folder. Also listed under 'Wan 2.1 Models' and 'LoRA Models'. See SwarmUI Video Docs for usage details."
}

wan_self_forcing_lora_entry = {
    "name": "Wan 2.1 14B Self Forcing LoRA T2V/I2V",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan21_14B_Self_Forcing_LoRA_T2V_I2V.safetensors",
    "save_filename": "Wan21_14B_Self_Forcing_LoRA_T2V_I2V.safetensors",
    "target_dir_key": "Lora",
    "info": "Self Forcing LoRA for Wan 2.1 14B T2V/I2V models. Saves to Lora folder. See SwarmUI Video Docs for usage details."
}

# Define the new model entries
wan_lightx2v_lora_entry = {
    "name": "Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64_fixed.safetensors",
    "save_filename": "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64_fixed.safetensors",
    "target_dir_key": "Lora",
    "info": "LightX2V CFG Step Distill LoRA V2 for Wan 2.1 14B T2V and I2V models. Saves to Lora folder. See SwarmUI Video Docs for usage details."
}

wan_uni3c_controlnet_lora_entry = {
    "name": "Wan 2.1 Uni3C ControlNet",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan21_Uni3C_controlnet_fp16.safetensors",
    "save_filename": "Wan21_Uni3C_controlnet_fp16.safetensors",
    "target_dir_key": "controlnet",
    "info": "Uni3C ControlNet for Wan 2.1 models. Saves to ControlNet folder."
}

wan_vae_entry = {
    "name": "Wan 2.1 VAE BF16",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan2_1_VAE_bf16.safetensors",
    "save_filename": "Wan2_1_VAE_bf16.safetensors",
    "target_dir_key": "vae"
}

# Define the new FusionX I2V GGUF models
wan_fusionx_i2v_gguf_q4_entry = {
    "name": "Wan 2.1 FusionX I2V 14B GGUF Q4_K_M",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q4_K_M.gguf",
    "save_filename": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q4_K_M.gguf",
    "target_dir_key": "diffusion_models"
}

wan_fusionx_i2v_gguf_q5_entry = {
    "name": "Wan 2.1 FusionX I2V 14B GGUF Q5_K_M",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q5_K_M.gguf",
    "save_filename": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q5_K_M.gguf",
    "target_dir_key": "diffusion_models"
}

wan_fusionx_i2v_gguf_q6_entry = {
    "name": "Wan 2.1 FusionX I2V 14B GGUF Q6_K",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q6_K.gguf",
    "save_filename": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q6_K.gguf",
    "target_dir_key": "diffusion_models"
}

wan_fusionx_i2v_gguf_q8_entry = {
    "name": "Wan 2.1 FusionX I2V 14B GGUF Q8",
    "repo_id": "MonsterMMORPG/Wan_GGUF",
    "filename_in_repo": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q8.gguf",
    "save_filename": "Wan2.1_14b_FusionX_Image_to_Video_GGUF_Q8.gguf",
    "target_dir_key": "diffusion_models"
}

# Define the new FusionX T2V GGUF models
wan_fusionx_t2v_gguf_q4_entry = {
    "name": "Wan 2.1 FusionX T2V 14B GGUF Q4_K_M",
    "repo_id": "QuantStack/Wan2.1_T2V_14B_FusionX-GGUF",
    "filename_in_repo": "Wan2.1_T2V_14B_FusionX-Q4_K_M.gguf",
    "save_filename": "Wan2.1_T2V_14B_FusionX_GGUF_Q4_K_M.gguf",
    "target_dir_key": "diffusion_models"
}

wan_fusionx_t2v_gguf_q5_entry = {
    "name": "Wan 2.1 FusionX T2V 14B GGUF Q5_K_M",
    "repo_id": "QuantStack/Wan2.1_T2V_14B_FusionX-GGUF",
    "filename_in_repo": "Wan2.1_T2V_14B_FusionX-Q5_K_M.gguf",
    "save_filename": "Wan2.1_T2V_14B_FusionX_GGUF_Q5_K_M.gguf",
    "target_dir_key": "diffusion_models"
}

wan_fusionx_t2v_gguf_q6_entry = {
    "name": "Wan 2.1 FusionX T2V 14B GGUF Q6_K",
    "repo_id": "QuantStack/Wan2.1_T2V_14B_FusionX-GGUF",
    "filename_in_repo": "Wan2.1_T2V_14B_FusionX-Q6_K.gguf",
    "save_filename": "Wan2.1_T2V_14B_FusionX_GGUF_Q6_K.gguf",
    "target_dir_key": "diffusion_models"
}

wan_fusionx_t2v_gguf_q8_entry = {
    "name": "Wan 2.1 FusionX T2V 14B GGUF Q8_0",
    "repo_id": "QuantStack/Wan2.1_T2V_14B_FusionX-GGUF",
    "filename_in_repo": "Wan2.1_T2V_14B_FusionX-Q8_0.gguf",
    "save_filename": "Wan2.1_T2V_14B_FusionX_GGUF_Q8_0.gguf",
    "target_dir_key": "diffusion_models"
}

models_structure = {
    "SwarmUI Bundles": {
        "info": "Download pre-defined bundles of commonly used models for SwarmUI with a single click.",
        "bundles": [
            {
                "name": "Qwen Image Core Bundle",
                "info": (
                    "Downloads the core Qwen Image models for image generation with necessary components.\n\n"
                    "**Includes:**\n"
                    "- Qwen_Image_Q8_0 (High quality GGUF model)\n"
                    "- Qwen_Image_Edit_GGUF_Q8_0 (High quality GGUF image editing model)\n"
                    "- qwen_2.5_vl_7b_fp8_scaled.safetensors (Text encoder)\n"
                    "- qwen_image_vae.safetensors (VAE model)\n"
                    "- Qwen Image Lightning 8steps V1.1 LoRA (Fast inference LoRA)\n"
                ),
                "models_to_download": [
                    ("Image Generation Models", "Qwen Image Models", "Qwen_Image_Q8_0"),
                    ("Image Generation Models", "Qwen Image Editing Models", "Qwen_Image_Edit_GGUF_Q8_0"),
                    ("Text Encoder Models", "Clip Models", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
                    ("VAE Models", "Most Common VAEs (e.g. FLUX and HiDream-I1)", "qwen_image_vae.safetensors"),
                    ("Image Generation Models", "Qwen Image Models", "Qwen Image Lightning 8steps V1.1 LoRA"),
                ]
            },
            {
                "name": "Wan 2.2 Core 8 Steps Bundle",
                "info": (
                    "Downloads the new Wan 2.2 models in FP8 precision for efficient 8-step video generation, with all necessary supporting files.\n\n"
                    "**Includes:**\n"
                    "- Wan 2.2 I2V High Noise 14B FP8 Scaled\n"
                    "- Wan 2.2 I2V Low Noise 14B FP8 Scaled\n"
                    "- Wan 2.2 T2V High Noise 14B FP8 Scaled\n"
                    "- Wan 2.2 T2V Low Noise 14B FP8 Scaled\n"
                    "- Wan 2.2 VAE\n"
                    "- UMT5 XXL FP16 (Default for SwarmUI)\n"
                    "- Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_Low\n"
                    "- Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_High\n"
                    "- Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low\n"
                    "- Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High\n"
                    "- Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)\n"
                    "\n"
                    "**How to use Wan 2.2:** [Wan 2.2 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-22-parameters)"
                ),
                "models_to_download": [
                    ("Video Generation Models", "Wan 2.2 Official Models", "Wan 2.2 I2V High Noise 14B FP8 Scaled"),
                    ("Video Generation Models", "Wan 2.2 Official Models", "Wan 2.2 I2V Low Noise 14B FP8 Scaled"),
                    ("Video Generation Models", "Wan 2.2 Official Models", "Wan 2.2 T2V High Noise 14B FP8 Scaled"),
                    ("Video Generation Models", "Wan 2.2 Official Models", "Wan 2.2 T2V Low Noise 14B FP8 Scaled"),
                    ("VAE Models", "Most Common VAEs (e.g. FLUX and HiDream-I1)", "Wan 2.2 VAE"),
                    ("Text Encoder Models", "UMT5 XXL Models", "UMT5 XXL FP16 (Save As default for SwarmUI)"),
                    ("Video Generation Models", "Wan 2.2 LoRAs", "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_Low"),
                    ("Video Generation Models", "Wan 2.2 LoRAs", "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_High"),
                    ("Video Generation Models", "Wan 2.2 LoRAs", "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low"),
                    ("Video Generation Models", "Wan 2.2 LoRAs", "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High"),
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)"),
                ]
            },
            {
                "name": "Wan 2.1 Core Models Bundle (GGUF Q6_K + Best LoRAs)",
                "info": (
                    "Downloads a core set of Wan 2.1 models for video generation, including T2V, I2V, and companion LoRAs, plus the recommended UMT5 text encoder and CLIP Vision H.\n\n"
                    "**Includes:**\n"
                    "- Wan 2.1 T2V 1.3B FP16\n"
                    "- Wan 2.1 T2V 14B FusionX LoRA\n"
                    "- Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)\n"
                    "- Phantom Wan 14B FusionX LoRA\n"
                    "- Wan 2.1 I2V 14B FusionX LoRA\n"
                    "- Wan 2.1 14B Self Forcing LoRA T2V/I2V\n"
                    "- Wan 2.1 T2V 14B 720p GGUF Q6_K\n"
                    "- Wan 2.1 I2V 14B 720p GGUF Q6_K\n"
                    "- UMT5 XXL FP8 Scaled (Default for SwarmUI)\n"
                    "- CLIP Vision H (Used by Wan 2.1)\n"
                    "\n"
                    "**How to use Wan 2.1:** [Wan 2.1 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21-parameters)"
                ),
                "models_to_download": [
                    ("Video Generation Models", "Wan 2.1 Official Models", "Wan 2.1 T2V 1.3B FP16"),
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Wan 2.1 T2V 14B FusionX LoRA"),
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)"),
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Phantom Wan 14B FusionX LoRA"),
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Wan 2.1 I2V 14B FusionX LoRA"),
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Wan 2.1 14B Self Forcing LoRA T2V/I2V"),
                    ("Video Generation Models", "Wan 2.1 Official Models", "Wan 2.1 T2V 14B 720p GGUF Q6_K"),
                    ("Video Generation Models", "Wan 2.1 Official Models", "Wan 2.1 I2V 14B 720p GGUF Q6_K"),
                    ("Text Encoder Models", "UMT5 XXL Models", "UMT5 XXL FP8 Scaled (Default for SwarmUI)"),
                    ("Clip Vision Models", "Standard Clip Vision Models", "CLIP Vision H (Used by Wan 2.1)"),
                ]
            },
            {
                "name": "FLUX Models Bundle",
                "info": (
                    "Downloads a core set of models for using FLUX models in SwarmUI, plus common utility models.\n\n"
                    "**Includes:**\n"
                    "- FLUX Kontext DEV FP16 (Saved as FLUX_Kontext_Dev.safetensors)\n"
                    "- FLUX DEV 1.0 FP16 (Saved as FLUX_Dev.safetensors)\n"
                    "- FLUX DEV Fill (In/Out-Painting) (Saved as FLUX_DEV_Fill.safetensors)\n"
                    "- FLUX DEV Redux (Style/Mix) (Saved as FLUX_DEV_Redux.safetensors)\n" 
                    "- FLUX Krea DEV (Saved as FLUX_Krea_Dev.safetensors)\n"
                    "- T5 XXL FP16 (Saved as t5xxl_enconly.safetensors)\n"
                    "- FLUX VAE (Saved as FLUX_VAE.safetensors)\n"
                    "- CLIP-SAE-ViT-L-14 (Saved as clip_l.safetensors - SwarmUI Default)\n"
                    "- Best Image Upscaler Models (Full Set)\n"
                    "- Face Segment/Masking Models (Full Set)\n"
                    "\n"
                    "**How to use FLUX:** [FLUX Model Support](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#black-forest-labs-flux1-models)\n"
                    "**Important Setup Guide:** [General FLUX Install/Usage](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#install)"
                ),
                "models_to_download": [
                    ("Image Generation Models", "FLUX Models", "FLUX Kontext DEV FP16"),
                    ("Image Generation Models", "FLUX Models", "FLUX DEV 1.0 FP16"),
                    ("Image Generation Models", "FLUX Models", "FLUX DEV Fill (In/Out-Painting)"),
                    ("Image Generation Models", "FLUX Models", "FLUX DEV Redux (Style/Mix)"),
                    ("Image Generation Models", "FLUX Models", "FLUX Krea DEV"),
                    ("Text Encoder Models", "T5 XXL Models", "T5 XXL FP16 (Save As t5xxl_enconly for SwarmUI default name)"),
                    ("VAE Models", "Most Common VAEs (e.g. FLUX and HiDream-I1)", "FLUX VAE as FLUX_VAE.safetensors (Used by FLUX, HiDream, etc.)"),
                    ("Text Encoder Models", "Clip Models", "CLIP-SAE-ViT-L-14 (Save As clip_l.safetensors - SwarmUI default name)"),
                    ("Other Models (e.g. Yolo Face Segment, Image Upscaling)", "Image Upscaling Models", "Best Upscaler Models (Full Set Snapshot)"),
                    ("Other Models (e.g. Yolo Face Segment, Image Upscaling)", "Auto Yolo Masking/Segment Models", "Face Segment/Masking Models (Full Set Snapshot)"),
                ]
            },
            {
                 "name": "HiDream-I1 Dev Bundle (Recommended)",
                 "info": (
                     "Downloads the recommended HiDream-I1 Dev model (Q8 GGUF), necessary supporting files, and common utility models.\n\n"
                     "**Includes:**\n"
                     "- HiDream-I1 Dev GGUF Q8_0 (Saved as HiDream_I1_Dev_GGUF_Q8_0.gguf)\n"
                     "- T5 XXL FP16 (Saved as t5xxl_enconly.safetensors)\n"
                     "- Long Clip L for HiDream-I1 (Saved as long_clip_l_hi_dream.safetensors)\n"
                     "- Long Clip G for HiDream-I1 (Saved as long_clip_g_hi_dream.safetensors)\n"
                     "- LLAMA 3.1 8b Instruct FP8 Scaled for HiDream-I1 (Saved as llama_3.1_8b_instruct_fp8_scaled.safetensors)\n"
                     "- FLUX VAE (Saved as FLUX_VAE.safetensors)\n"
                     "- Best Image Upscaler Models (Full Set)\n"
                     "- Face Segment/Masking Models (Full Set)\n"
                     "\n"
                     "**How to use HiDream:** [HiDream Model Support](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#hidream-i1)"
                 ),
                 "models_to_download": [
                     ("Image Generation Models", "HiDream-I1 Dev Models (Recommended)", "HiDream-I1 Dev GGUF Q8_0"),
                     ("Text Encoder Models", "T5 XXL Models", "T5 XXL FP16 (Save As t5xxl_enconly for SwarmUI default name)"),
                     ("Text Encoder Models", "Clip Models", "Long Clip L for HiDream-I1"),
                     ("Text Encoder Models", "Clip Models", "Long Clip G for HiDream-I1"),
                     ("Text Encoder Models", "LLM Text Encoders", "LLAMA 3.1 8b Instruct FP8 Scaled for HiDream-I1"),
                     ("VAE Models", "Most Common VAEs (e.g. FLUX and HiDream-I1)", "FLUX VAE as FLUX_VAE.safetensors (Used by FLUX, HiDream, etc.)"),
                     ("Other Models (e.g. Yolo Face Segment, Image Upscaling)", "Image Upscaling Models", "Best Upscaler Models (Full Set Snapshot)"),
                     ("Other Models (e.g. Yolo Face Segment, Image Upscaling)", "Auto Yolo Masking/Segment Models", "Face Segment/Masking Models (Full Set Snapshot)"),
                 ]
             },
        ]
    },
    "ComfyUI Bundles": {
        "info": "Download pre-defined bundles for specific ComfyUI workflows, including models and related assets.",
        "bundles": [
            {
                "name": "Clothing Migration Workflow Bundle",
                "info": (
                    "Downloads all necessary models and assets for the Clothing Migration workflow in ComfyUI (SwarmUI backend).\n\n"
                    "**Includes:**\n"
                    "- Joy Caption Alpha Two (Captioning Assets)\n"
                    "- Migration LoRA Cloth (TTPlanet)\n"
                    "- Figures TTP Migration LoRA (TTPlanet)\n"
                    "- SigLIP SO400M Patch14 384px (Full Repo)\n"
                    "- Meta-Llama-3.1-8B-Instruct (Full Repo)\n"
                    "- FLUX VAE (Standard VAE, saved as ae.safetensors)\n"
                    "- FLUX DEV ControlNet Inpainting Beta (Alimama) (ControlNet for inpainting)\n"
                    "- T5 XXL FP16 (Text Encoder)\n"
                    "- CLIP-SAE-ViT-L-14 (CLIP L Text Encoder, saved as clip_l.safetensors)\n"
                    "\n"
                    "**Important:** Ensure your ComfyUI setup and the specific workflow are configured to use these models in their respective SwarmUI model paths. "
                    "This bundle downloads models to their default SwarmUI locations (e.g., Models/Lora, Models/LLM, Models/controlnet, etc.)."
                ),
                "models_to_download": [
                    ("ComfyUI Workflows", "Captioning Workflows", "Joy Caption Alpha Two (Full Repo)"),
                    ("LoRA Models", "Various LoRAs", "Migration LoRA Cloth (TTPlanet)"),
                    ("LoRA Models", "Various LoRAs", "Figures TTP Migration LoRA (TTPlanet)"),
                    ("Clip Vision Models", "SigLIP Vision Models", "SigLIP SO400M Patch14 384px (Full Repo)"),
                    ("LLM Models", "General LLMs", "Meta-Llama-3.1-8B-Instruct (Full Repo)"),
                    ("VAE Models", "Most Common VAEs (e.g. FLUX and HiDream-I1)", "FLUX VAE as ae.safetensors"),
                    ("Image Generation Models", "FLUX Models", "FLUX DEV ControlNet Inpainting Beta (Alimama)"),
                    ("Text Encoder Models", "T5 XXL Models", "T5 XXL FP16"),
                    ("Text Encoder Models", "Clip Models", "CLIP-SAE-ViT-L-14 (Save As clip_l.safetensors - SwarmUI default name)"),
                ]
            },
            {
                "name": "ComfyUI MultiTalk Bundle",
                "info": (
                    "Downloads all necessary models for ComfyUI MultiTalk workflow, including the latest Wan 2.1 MultiTalk model and supporting components.\n\n"
                    "**Includes:**\n"
                    "- Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)\n"
                    "- Wan 2.1 Uni3C ControlNet\n"
                    "- WanVideo 2.1 MultiTalk 14B FP32\n"
                    "- Wan 2.1 I2V 14B 480p GGUF Q8\n"
                    "- Wan 2.1 I2V 14B 720p GGUF Q8\n"
                    "- Wan 2.1 FusionX I2V 14B GGUF Q8\n"
                    "- CLIP Vision H (Used by Wan 2.1)\n"
                    "- UMT5 XXL FP16 (Default for SwarmUI)\n"
                    "- Wan 2.1 VAE BF16\n"
                    "\n"
                    "**How to use Wan 2.1:** [Wan 2.1 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21-parameters)"
                ),
                "models_to_download": [
                    ("Video Generation Models", "Wan 2.1 LoRAs", "Wan 2.1 14B LightX2V CFG Step Distill LoRA V2 (T2V + I2V) (Rank 64)"),
                    ("ControlNet Models", "Various ControlNets", "Wan 2.1 Uni3C ControlNet"),
                    ("Video Generation Models", "Wan 2.1 Official Models", "WanVideo 2.1 MultiTalk 14B FP32"),
                    ("Video Generation Models", "Wan 2.1 Official Models", "Wan 2.1 I2V 14B 480p GGUF Q8"),
                    ("Video Generation Models", "Wan 2.1 Official Models", "Wan 2.1 I2V 14B 720p GGUF Q8"),
                    ("Video Generation Models", "Wan 2.1 FusionX Models", "Wan 2.1 FusionX I2V 14B GGUF Q8"),
                    ("Clip Vision Models", "Standard Clip Vision Models", "CLIP Vision H (Used by Wan 2.1)"),
                    ("Text Encoder Models", "UMT5 XXL Models", "UMT5 XXL FP16 (Save As default for SwarmUI)"),
                    ("VAE Models", "Most Common VAEs (e.g. FLUX and HiDream-I1)", "Wan 2.1 VAE BF16"),
                ]
            },
        ]
    },
    "Image Generation Models": {
        "info": "Models for generating images from text or other inputs.",
        "sub_categories": {
            "Qwen Image Models": {
                "info": "Qwen Image generation models in various quantization formats (GGUF and safetensors).",
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "Qwen_Image_Q4_1", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen-image-Q4_1.gguf", "save_filename": "Qwen_Image_Q4_1.gguf"},
                    {"name": "Qwen_Image_Q5_1", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen-image-Q5_1.gguf", "save_filename": "Qwen_Image_Q5_1.gguf"},
                    {"name": "Qwen_Image_Q6_K", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen-image-Q6_K.gguf", "save_filename": "Qwen_Image_Q6_K.gguf"},
                    {"name": "Qwen_Image_Q8_0", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen-image-Q8_0.gguf", "save_filename": "Qwen_Image_Q8_0.gguf"},
                    {"name": "Qwen_Image_FP8_e4m3f", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen_image_fp8_e4m3fn.safetensors", "save_filename": "qwen_image_fp8_e4m3fn.safetensors"},
                    {"name": "Qwen_Image_BF16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen_image_bf16.safetensors", "save_filename": "qwen_image_bf16.safetensors"},
                    {"name": "Qwen Image Lightning 8steps V1.1 LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen-Image-Lightning-8steps-V1.1.safetensors", "save_filename": "Qwen-Image-Lightning-8steps-V1.1.safetensors", "target_dir_key": "Lora", "info": "Qwen Image Lightning LoRA for fast 8-step image generation. Saves to Lora folder. Use with Qwen Image models for optimized inference."},
                ]
            },
            "Qwen Image Editing Models": {
                "info": "Qwen Image editing models in various quantization formats (GGUF and safetensors) for image editing tasks.",
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "Qwen_Image_Edit_BF16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen_Image_Edit_BF16.safetensors", "save_filename": "Qwen_Image_Edit_BF16.safetensors"},
                    {"name": "Qwen_Image_Edit_FP8_e4m3fn", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen_Image_Edit_FP8_e4m3fn.safetensors", "save_filename": "Qwen_Image_Edit_FP8_e4m3fn.safetensors"},
                    {"name": "Qwen_Image_Edit_GGUF_Q8_0", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen_Image_Edit_GGUF_Q8_0.gguf", "save_filename": "Qwen_Image_Edit_GGUF_Q8_0.gguf"},
                    {"name": "Qwen_Image_Edit_GGUF_Q6_K", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen_Image_Edit_GGUF_Q6_K.gguf", "save_filename": "Qwen_Image_Edit_GGUF_Q6_K.gguf"},
                    {"name": "Qwen_Image_Edit_GGUF_Q5_1", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen_Image_Edit_GGUF_Q5_1.gguf", "save_filename": "Qwen_Image_Edit_GGUF_Q5_1.gguf"},
                    {"name": "Qwen_Image_Edit_GGUF_Q4_K_M", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Qwen_Image_Edit_GGUF_Q4_K_M.gguf", "save_filename": "Qwen_Image_Edit_GGUF_Q4_K_M.gguf"},
                ]
            },
            "FLUX Models": {
                "info": ("FLUX models including Dev, ControlNet-like variants in standard formats (safetensors, FP16, FP8).\n\n"
                         "**How to use FLUX:** [FLUX Model Support](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#black-forest-labs-flux1-models)\n"
                         "**Extremely Important How To Use Parameters and Guide:**\n"
                         "- [General FLUX Install/Usage](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#install)\n"
                         "- [FLUX Tools Usage (Depth, Canny, etc.)](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#flux1-tools)"),
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "FLUX Krea DEV", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "flux1-krea-dev.safetensors", "save_filename": "FLUX_Krea_Dev.safetensors"},
                    {"name": "FLUX Kontext DEV FP16", "repo_id": "MonsterMMORPG/Best_FLUX_Models", "filename_in_repo": "flux1-kontext-dev.safetensors", "save_filename": "FLUX_Kontext_Dev.safetensors"},
                    {"name": "FLUX DEV 1.0 FP16", "repo_id": "OwlMaster/FLUX_LoRA_Train", "filename_in_repo": "flux1-dev.safetensors", "save_filename": "FLUX_Dev.safetensors"},
                    {"name": "FLUX DEV Fill (In/Out-Painting)", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "flux1-fill-dev.safetensors", "save_filename": "FLUX_DEV_Fill.safetensors"},
                    {"name": "FLUX DEV Depth", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "flux1-depth-dev.safetensors", "save_filename": "FLUX_DEV_Depth.safetensors"},
                    {"name": "FLUX DEV Canny", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "flux1-canny-dev.safetensors", "save_filename": "FLUX_DEV_Canny.safetensors"},
                    {"name": "FLUX DEV Redux (Style/Mix)", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "flux1-redux-dev.safetensors", "save_filename": "FLUX_DEV_Redux.safetensors", "target_dir_key": "style_models"},
                    {"name": "FLUX DEV 1.0 FP8 Scaled", "repo_id": "comfyanonymous/flux_dev_scaled_fp8_test", "filename_in_repo": "flux_dev_fp8_scaled_diffusion_model.safetensors", "save_filename": "FLUX_Dev_FP8_Scaled.safetensors"},
                    {"name": "FLUX DEV PixelWave V3", "repo_id": "mikeyandfriends/PixelWave_FLUX.1-dev_03", "filename_in_repo": "pixelwave_flux1_dev_bf16_03.safetensors", "save_filename": "FLUX_DEV_PixelWave_V3.safetensors"},
                    {"name": "FLUX DEV De-Distilled (Normal CFG 3.5)", "repo_id": "nyanko7/flux-dev-de-distill", "filename_in_repo": "consolidated_s6700.safetensors", "save_filename": "FLUX_DEV_De_Distilled.safetensors"},
                    {"name": "Flux Sigma Vision Alpha1 FP16 (Normal CFG 3.5)", "repo_id": "MonsterMMORPG/Best_FLUX_Models", "filename_in_repo": "fluxSigmaVision_fp16.safetensors", "save_filename": "Flux_Sigma_Vision_Alpha1_FP16.safetensors"},
                    {"name": "FLEX 1 Alpha (New Arch)", "repo_id": "ostris/Flex.1-alpha", "filename_in_repo": "Flex.1-alpha.safetensors", "save_filename": "FLEX_1_Alpha.safetensors"},
                    {
                        "name": "FLUX DEV ControlNet Inpainting Beta (Alimama)",
                        "repo_id": "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
                        "filename_in_repo": "diffusion_pytorch_model.safetensors",
                        "save_filename": "alimama_flux_inpainting.safetensors",
                        "target_dir_key": "controlnet"
                    },
                ]
            },
            "FLUX GGUF Models": {
                "info": ("FLUX models in GGUF quantized format for reduced memory usage. Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).\n\n"
                         "**How to use FLUX:** [FLUX Model Support](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#black-forest-labs-flux1-models)"),
                "target_dir_key": "diffusion_models",
                "models": [
                    # FLUX Krea DEV GGUF - Q8 to Q4
                    {"name": "FLUX Krea DEV GGUF Q8_0", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "flux1_krea_dev_BF16_Q8_0.gguf", "save_filename": "FLUX_Krea_Dev_GGUF_Q8_0.gguf"},
                    {"name": "FLUX Krea DEV GGUF Q6_K", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "flux1_krea_dev_BF16_Q6_K.gguf", "save_filename": "FLUX_Krea_Dev_GGUF_Q6_K.gguf"},
                    {"name": "FLUX Krea DEV GGUF Q5_1", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "flux1_krea_dev_BF16_Q5_1.gguf", "save_filename": "FLUX_Krea_Dev_GGUF_Q5_1.gguf"},
                    {"name": "FLUX Krea DEV GGUF Q4_1", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "flux1_krea_dev_BF16_Q4_1.gguf", "save_filename": "FLUX_Krea_Dev_GGUF_Q4_1.gguf"},
                    # FLUX DEV 1.0 GGUF - Q8 to Q4
                    {"name": "FLUX DEV 1.0 GGUF Q8", "repo_id": "city96/FLUX.1-dev-gguf", "filename_in_repo": "flux1-dev-Q8_0.gguf", "save_filename": "FLUX_Dev_GGUF_Q8.gguf"},
                    {"name": "FLUX DEV 1.0 GGUF Q6_K", "repo_id": "city96/FLUX.1-dev-gguf", "filename_in_repo": "flux1-dev-Q6_K.gguf", "save_filename": "FLUX_Dev_GGUF_Q6_K.gguf"},
                    {"name": "FLUX DEV 1.0 GGUF Q5_K_S", "repo_id": "city96/FLUX.1-dev-gguf", "filename_in_repo": "flux1-dev-Q5_K_S.gguf", "save_filename": "FLUX_Dev_GGUF_Q5_K_S.gguf"},
                    {"name": "FLUX DEV 1.0 GGUF Q4_K_S", "repo_id": "city96/FLUX.1-dev-gguf", "filename_in_repo": "flux1-dev-Q4_K_S.gguf", "save_filename": "FLUX_Dev_GGUF_Q4_K_S.gguf"},
                    # FLUX DEV Fill GGUF - Q8 to Q4
                    {"name": "FLUX DEV Fill GGUF Q8_0", "repo_id": "YarvixPA/FLUX.1-Fill-dev-gguf", "filename_in_repo": "flux1-fill-dev-Q8_0.gguf", "save_filename": "FLUX_DEV_Fill_GGUF_Q8_0.gguf"},
                    {"name": "FLUX DEV Fill GGUF Q6_K", "repo_id": "YarvixPA/FLUX.1-Fill-dev-gguf", "filename_in_repo": "flux1-fill-dev-Q6_K.gguf", "save_filename": "FLUX_DEV_Fill_GGUF_Q6_K.gguf"},
                    {"name": "FLUX DEV Fill GGUF Q5_K_S", "repo_id": "YarvixPA/FLUX.1-Fill-dev-gguf", "filename_in_repo": "flux1-fill-dev-Q5_K_S.gguf", "save_filename": "FLUX_DEV_Fill_GGUF_Q5_K_S.gguf"},
                    {"name": "FLUX DEV Fill GGUF Q4_K_S", "repo_id": "YarvixPA/FLUX.1-Fill-dev-gguf", "filename_in_repo": "flux1-fill-dev-Q4_K_S.gguf", "save_filename": "FLUX_DEV_Fill_GGUF_Q4_K_S.gguf"},
                    # FLUX Kontext DEV GGUF - Q8 to Q4
                    {"name": "FLUX Kontext DEV GGUF Q8_0", "repo_id": "bullerwins/FLUX.1-Kontext-dev-GGUF", "filename_in_repo": "flux1-kontext-dev-Q8_0.gguf", "save_filename": "FLUX_Kontext_Dev_GGUF_Q8_0.gguf"},
                    {"name": "FLUX Kontext DEV GGUF Q6_K", "repo_id": "bullerwins/FLUX.1-Kontext-dev-GGUF", "filename_in_repo": "flux1-kontext-dev-Q6_K.gguf", "save_filename": "FLUX_Kontext_Dev_GGUF_Q6_K.gguf"},
                    {"name": "FLUX Kontext DEV GGUF Q5_K_M", "repo_id": "bullerwins/FLUX.1-Kontext-dev-GGUF", "filename_in_repo": "flux1-kontext-dev-Q5_K_M.gguf", "save_filename": "FLUX_Kontext_Dev_GGUF_Q5_K_M.gguf"},
                    {"name": "FLUX Kontext DEV GGUF Q4_K_M", "repo_id": "bullerwins/FLUX.1-Kontext-dev-GGUF", "filename_in_repo": "flux1-kontext-dev-Q4_K_M.gguf", "save_filename": "FLUX_Kontext_Dev_GGUF_Q4_K_M.gguf"},
                ]
            },
            "HiDream-I1 Image Editing Models": {
                "info": f"Image editing specific variant of HiDream-I1.\n\n**How to use HiDream:** [{HIDREAM_INFO_LINK}]({HIDREAM_INFO_LINK})",
                "target_dir_key": "diffusion_models",
                "models": [
                     {"name": "HiDream-I1-E1 BF16 Image Editing", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_e1_full_bf16.safetensors", "save_filename": "HiDream_I1_E1_Image_Editing_BF16.safetensors"},
                ]
            },
            "HiDream-I1 Full Models": {
                "info": f"Full version of HiDream-I1 models. {GGUF_QUALITY_INFO}\n\n**How to use HiDream:** [{HIDREAM_INFO_LINK}]({HIDREAM_INFO_LINK})",
                "target_dir_key": "diffusion_models",
                "models": [
                     {"name": "HiDream-I1 Full FP16", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_i1_full_fp16.safetensors", "save_filename": "HiDream_I1_Full_FP16.safetensors"},
                     {"name": "HiDream-I1 Full FP8", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_i1_full_fp8.safetensors", "save_filename": "HiDream_I1_Full_FP8.safetensors"},
                     {"name": "HiDream-I1 Full GGUF F16", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-F16.gguf", "save_filename": "HiDream_I1_Full_GGUF_F16.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q8_0", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q8_0.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q8_0.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q6_K", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q6_K.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q6_K.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q5_K_M", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q5_K_M.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q5_K_M.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q5_K_S", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q5_K_S.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q5_K_S.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q5_1", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q5_1.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q5_1.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q5_0", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q5_0.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q5_0.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q4_K_M", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q4_K_M.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q4_K_M.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q4_K_S", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q4_K_S.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q4_K_S.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q4_1", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q4_1.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q4_1.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q4_0", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q4_0.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q4_0.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q3_K_M", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q3_K_M.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q3_K_M.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q3_K_S", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q3_K_S.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q3_K_S.gguf"},
                     {"name": "HiDream-I1 Full GGUF Q2_K", "repo_id": "city96/HiDream-I1-Full-gguf", "filename_in_repo": "hidream-i1-full-Q2_K.gguf", "save_filename": "HiDream_I1_Full_GGUF_Q2_K.gguf"},
                ]
            },
            "HiDream-I1 Dev Models (Recommended)": {
                "info": f"Development version of HiDream-I1 models (Recommended for general use). {GGUF_QUALITY_INFO}\n\n**How to use HiDream:** [{HIDREAM_INFO_LINK}]({HIDREAM_INFO_LINK})",
                "target_dir_key": "diffusion_models",
                "models": [
                     {"name": "HiDream-I1 Dev BF16", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_i1_dev_bf16.safetensors", "save_filename": "HiDream_I1_Dev_BF16.safetensors"},
                     {"name": "HiDream-I1 Dev FP8", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_i1_dev_fp8.safetensors", "save_filename": "HiDream_I1_Dev_FP8.safetensors"},
                     {"name": "HiDream-I1 Dev GGUF BF16", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-BF16.gguf", "save_filename": "HiDream_I1_Dev_GGUF_BF16.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q8_0", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q8_0.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q8_0.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q6_K", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q6_K.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q6_K.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q5_K_M", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q5_K_M.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q5_K_M.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q5_K_S", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q5_K_S.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q5_K_S.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q5_1", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q5_1.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q5_1.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q5_0", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q5_0.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q5_0.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q4_K_M", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q4_K_M.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q4_K_M.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q4_K_S", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q4_K_S.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q4_K_S.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q4_1", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q4_1.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q4_1.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q4_0", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q4_0.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q4_0.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q3_K_M", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q3_K_M.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q3_K_M.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q3_K_S", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q3_K_S.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q3_K_S.gguf"},
                     {"name": "HiDream-I1 Dev GGUF Q2_K", "repo_id": "city96/HiDream-I1-Dev-gguf", "filename_in_repo": "hidream-i1-dev-Q2_K.gguf", "save_filename": "HiDream_I1_Dev_GGUF_Q2_K.gguf"},
                ]
            },
            "HiDream-I1 Fast Models": {
                "info": f"Faster distilled version of HiDream-I1 models. {GGUF_QUALITY_INFO}\n\n**How to use HiDream:** [{HIDREAM_INFO_LINK}]({HIDREAM_INFO_LINK})",
                "target_dir_key": "diffusion_models",
                "models": [
                     {"name": "HiDream-I1 Fast BF16", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_i1_fast_bf16.safetensors", "save_filename": "HiDream_I1_Fast_BF16.safetensors"},
                     {"name": "HiDream-I1 Fast FP8", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/diffusion_models/hidream_i1_fast_fp8.safetensors", "save_filename": "HiDream_I1_Fast_FP8.safetensors"},
                     {"name": "HiDream-I1 Fast GGUF BF16", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-BF16.gguf", "save_filename": "HiDream_I1_Fast_GGUF_BF16.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q8_0", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q8_0.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q8_0.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q6_K", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q6_K.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q6_K.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q5_K_M", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q5_K_M.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q5_K_M.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q5_K_S", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q5_K_S.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q5_K_S.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q5_1", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q5_1.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q5_1.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q5_0", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q5_0.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q5_0.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q4_K_M", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q4_K_M.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q4_K_M.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q4_K_S", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q4_K_S.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q4_K_S.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q4_1", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q4_1.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q4_1.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q4_0", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q4_0.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q4_0.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q3_K_M", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q3_K_M.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q3_K_M.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q3_K_S", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q3_K_S.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q3_K_S.gguf"},
                     {"name": "HiDream-I1 Fast GGUF Q2_K", "repo_id": "city96/HiDream-I1-Fast-gguf", "filename_in_repo": "hidream-i1-fast-Q2_K.gguf", "save_filename": "HiDream_I1_Fast_GGUF_Q2_K.gguf"},
                ]
            },
            "Stable Diffusion 1.5 Models": {
                 "info": "Popular fine-tuned models based on Stable Diffusion 1.5.",
                 "target_dir_key": "Stable-Diffusion",
                 "models": [
                    {"name": "Realistic Vision V6", "repo_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE", "filename_in_repo": "Realistic_Vision_V6.0_NV_B1.safetensors", "save_filename": "SD1.5_Realistic_Vision_V6.safetensors"},
                    {"name": "RealCartoon3D V18", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "realcartoon3dv18.safetensors", "save_filename": "SD1.5_RealCartoon3D_V18.safetensors"},
                    {"name": "CyberRealistic V8", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "cyberrealistic_v80.safetensors", "save_filename": "SD1.5_CyberRealistic_V8.safetensors"},
                    {"name": "epiCPhotoGasm Ultimate Fidelity", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "epicphotogasm_ultimateFidelity.safetensors", "save_filename": "epiCPhotoGasm_Ultimate_Fidelity.safetensors"},

                 ]
            },
            "Stable Diffusion XL (SDXL) Models": {
                 "info": "Models based on the Stable Diffusion XL architecture.",
                 "target_dir_key": "Stable-Diffusion",
                 "models": [
                    {"name": "SDXL Base 1.0 (Official)", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "sd_xl_base_1.0_0.9vae.safetensors", "save_filename": "SDXL_Base_1_0.safetensors"},
                    {"name": "Juggernaut XL V11", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "Juggernaut-XI-byRunDiffusion.safetensors", "save_filename": "SDXL_Juggernaut_V11.safetensors"},
                    {"name": "epiCRealism XL LastFame", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "epicrealismXL_vxviLastfameRealism.safetensors", "save_filename": "SDXL_epiCRealism_Last_LastFame.safetensors"},
                    {"name": "RealVisXL V5", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "realvisxlV50_v50Bakedvae.safetensors", "save_filename": "SDXL_RealVisXL_V5.safetensors"},
                    {"name": "Real Dream SDXL 5", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "realDream_sdxl5.safetensors", "save_filename": "SDXL_RealDream_5.safetensors"},
                    {"name": "Eldritch Photography V1", "repo_id": "OwlMaster/Some_best_SDXL", "filename_in_repo": "eldritchPhotography_v1.safetensors", "save_filename": "SDXL_Eldritch_Photography_V1.safetensors"},
                 ]
            },
            "Stable Diffusion 3.5 Large Models": {
                "info": "Official Stable Diffusion 3.5 Large models and variants. GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).",
                "target_dir_key": "diffusion_models",
                "models": [
                     {"name": "Stable Diffusion 3.5 Large (Official) - FP16", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "sd3.5_large.safetensors", "save_filename": "SD3.5_Official_Large.safetensors"},
                     {"name": "Stable Diffusion 3.5 Large (Official) - FP8 Scaled", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "sd3.5_large_fp8_scaled.safetensors", "save_filename": "SD3.5_Official_Large_FP8_Scaled.safetensors", "target_dir_key": "Stable-Diffusion"},
                     {"name": "Stable Diffusion 3.5 Large (Official) - GGUF Q8", "repo_id": "city96/stable-diffusion-3.5-large-gguf", "filename_in_repo": "sd3.5_large-Q8_0.gguf", "save_filename": "SD3.5_Official_Large_GGUF_Q8.gguf"},
                     {"name": "Stable Diffusion 3.5 Large (Official) - GGUF Q5_1", "repo_id": "city96/stable-diffusion-3.5-large-gguf", "filename_in_repo": "sd3.5_large-Q5_1.gguf", "save_filename": "SD3.5_Official_Large_GGUF_Q5_1.gguf"},
                     {"name": "Stable Diffusion 3.5 Large (Official) - GGUF Q4_1", "repo_id": "city96/stable-diffusion-3.5-large-gguf", "filename_in_repo": "sd3.5_large-Q4_1.gguf", "save_filename": "SD3.5_Official_Large_GGUF_Q4_1.gguf"},
                ]
            }
        }
    }, "Other Models (e.g. Yolo Face Segment, Image Upscaling)": {
        "info": "Utility models like upscalers and segmentation models.",
        "sub_categories": {
            "Image Upscaling Models": {
                "info": "High-quality deterministic image upscaling models (from OpenModelDB and other sources).",
                "target_dir_key": "upscale_models",
                "models": [
                    {"name": "Best Upscaler Models (Full Set Snapshot)", "repo_id": "MonsterMMORPG/BestImageUpscalers", "is_snapshot": True},
                    {"name": "LTX Spatial Upscaler 0.9.7 (Lightricks)", "repo_id": "Lightricks/LTX-Video", "filename_in_repo": "ltxv-spatial-upscaler-0.9.7.safetensors", "save_filename": "LTX_Spatial_Upscaler_0_9_7.safetensors"},
                    {"name": "LTX Temporal Upscaler 0.9.7 (Lightricks)", "repo_id": "Lightricks/LTX-Video", "filename_in_repo": "ltxv-temporal-upscaler-0.9.7.safetensors", "save_filename": "LTX_Temporal_Upscaler_0_9_7.safetensors"},
                ]
            },
            "Auto Yolo Masking/Segment Models": {
                 "info": "YOLO-based models for automatic face segmentation/masking (from MonsterMMORPG), useful for inpainting.",
                 "target_dir_key": "yolov8",
                 "models": [
                     {"name": "Face Segment/Masking Models (Full Set Snapshot)", "repo_id": "MonsterMMORPG/FaceSegments", "is_snapshot": True},
                 ]
             }
        }
    }, "Text Encoder Models": {
         "info": "Text encoder models used by various generation models.",
         "sub_categories": {
            "T5 XXL Models": {
                "info": "T5 XXL variants used by FLUX, SD 3.5, Hunyuan, etc. GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).",
                "target_dir_key": "clip",
                "models": [
                    {"name": "T5 XXL FP16", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "t5xxl_fp16.safetensors", "save_filename": "t5xxl_fp16.safetensors"},
                    {"name": "T5 XXL FP8 (e4m3fn)", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "t5xxl_fp8_e4m3fn.safetensors", "save_filename": "t5xxl_fp8_e4m3fn.safetensors"},
                    {"name": "T5 XXL FP8 Scaled (e4m3fn)", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "t5xxl_fp8_e4m3fn_scaled.safetensors", "save_filename": "t5xxl_fp8_e4m3fn_scaled.safetensors"},
                    {"name": "T5 XXL FP16 (Save As t5xxl_enconly for SwarmUI default name)", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "t5xxl_fp16.safetensors", "save_filename": "t5xxl_enconly.safetensors"},
                    {"name": "T5 XXL GGUF Q8", "repo_id": "calcuis/mochi", "filename_in_repo": "t5xxl_fp16-q8_0.gguf", "save_filename": "t5xxl_GGUF_Q8.gguf"},
                    {"name": "T5 XXL GGUF Q4_0", "repo_id": "calcuis/mochi", "filename_in_repo": "t5xxl_fp16-q4_0.gguf", "save_filename": "t5xxl_GGUF_Q4_0.gguf"},
                ]
            },
            "UMT5 XXL Models": {
                "info": "UMT5 XXL variants used by Wan 2.1. GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0). Select non-GGUF FP16/BF16/FP8 based on your Wan model choice, or use GGUF if preferred (manual setup needed in SwarmUI).",
                "target_dir_key": "clip",
                "models": [
                    {"name": "UMT5 XXL BF16 (Used by Wan 2.1)", "repo_id": "Kijai/WanVideo_comfy", "filename_in_repo": "umt5-xxl-enc-bf16.safetensors", "save_filename": "umt5-xxl-enc-bf16.safetensors"},
                    # These save the same file, choose one or rename target
                    {"name": "UMT5 XXL BF16 (Save As default for SwarmUI)", "repo_id": "Kijai/WanVideo_comfy", "filename_in_repo": "umt5-xxl-enc-bf16.safetensors", "save_filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "target_dir_key": "clip", "allow_overwrite": True},
                    {"name": "UMT5 XXL FP16 (Save As default for SwarmUI)", "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename_in_repo": "split_files/text_encoders/umt5_xxl_fp16.safetensors", "save_filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "target_dir_key": "clip", "allow_overwrite": True},
                    {"name": "UMT5 XXL FP8 Scaled (Default for SwarmUI)", "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename_in_repo": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "save_filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "target_dir_key": "clip", "allow_overwrite": True},
                    # GGUF models
                    {"name": "UMT5 XXL GGUF Q8 (Manual Setup)", "repo_id": "city96/umt5-xxl-encoder-gguf", "filename_in_repo": "umt5-xxl-encoder-Q8_0.gguf", "save_filename": "umt5-xxl-encoder-Q8_0.gguf"},
                    {"name": "UMT5 XXL GGUF Q6_K (Manual Setup)", "repo_id": "city96/umt5-xxl-encoder-gguf", "filename_in_repo": "umt5-xxl-encoder-Q6_K.gguf", "save_filename": "umt5-xxl-encoder-Q6_K.gguf"},
                    {"name": "UMT5 XXL GGUF Q5_K_M (Manual Setup)", "repo_id": "city96/umt5-xxl-encoder-gguf", "filename_in_repo": "umt5-xxl-encoder-Q5_K_M.gguf", "save_filename": "umt5-xxl-encoder-Q5_K_M.gguf"},
                    {"name": "UMT5 XXL GGUF Q4_K_M (Manual Setup)", "repo_id": "city96/umt5-xxl-encoder-gguf", "filename_in_repo": "umt5-xxl-encoder-Q4_K_M.gguf", "save_filename": "umt5-xxl-encoder-Q4_K_M.gguf"},
                ]
            },
            "Clip Models": {
                "info": "CLIP models (L and G variants) used by many models.",
                "target_dir_key": "clip",
                "models": [
                    {"name": "CLIP-SAE-ViT-L-14 (Save As clip_l.safetensors - SwarmUI default name)", "repo_id": "OwlMaster/zer0int-CLIP-SAE-ViT-L-14", "filename_in_repo": "clip_l.safetensors", "save_filename": "clip_l.safetensors", "pre_delete_target": True},
                    {"name": "CLIP-SAE-ViT-L-14 (Save As CLIP_SAE_ViT_L_14)", "repo_id": "OwlMaster/zer0int-CLIP-SAE-ViT-L-14", "filename_in_repo": "clip_l.safetensors", "save_filename": "CLIP_SAE_ViT_L_14.safetensors"},
                    {"name": "Default Clip L", "repo_id": "MonsterMMORPG/Kohya_Train", "filename_in_repo": "clip_l.safetensors", "save_filename": "clip_l.safetensors"}, # Use specific save name
                    {"name": "Clip G", "repo_id": "OwlMaster/SD3New", "filename_in_repo": "clip_g.safetensors", "save_filename": "clip_g.safetensors"},
                    {"name": "Long Clip L for HiDream-I1", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/text_encoders/clip_l_hidream.safetensors", "save_filename": "long_clip_l_hi_dream.safetensors"},
                    {"name": "Long Clip G for HiDream-I1", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/text_encoders/clip_g_hidream.safetensors", "save_filename": "long_clip_g_hi_dream.safetensors"},
                    {"name": "qwen_2.5_vl_7b_fp16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen_2.5_vl_7b_fp16.safetensors", "save_filename": "qwen_2.5_vl_7b_fp16.safetensors"},
                    {"name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "save_filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors"},
                ]
            },
            "LLM Text Encoders": {
                 "info": "Large Language Model based text encoders, currently used by HiDream-I1.",
                 "target_dir_key": "clip",
                 "models": [
                     {"name": "LLAMA 3.1 8b Instruct FP8 Scaled for HiDream-I1", "repo_id": "Comfy-Org/HiDream-I1_ComfyUI", "filename_in_repo": "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors", "save_filename": "llama_3.1_8b_instruct_fp8_scaled.safetensors"},
                 ]
             },
         }
    },
    "Video Generation Models": {
        "info": "Models for generating videos from text or images.",
        "sub_categories": {
            "Wan 2.1 Official Models": {
                 "info": ("Official Wan 2.1 text-to-video and image-to-video models (non-FusionX). GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).\n\n"
                          "**Extremely Important How To Use Parameters and Guide:** [Wan 2.1 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21-parameters)"),
                 "target_dir_key": "diffusion_models",
                 "models": [
                    {"name": "Wan 2.1 T2V 1.3B FP16", "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename_in_repo": "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", "save_filename": "Wan2.1_1.3b_Text_to_Video.safetensors"},
                    {"name": "Wan 2.1 T2V 14B 720p FP16", "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename_in_repo": "split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors", "save_filename": "Wan2.1_14b_Text_to_Video.safetensors"},
                    {"name": "Wan 2.1 T2V 14B 720p FP8", "repo_id": "Kijai/WanVideo_comfy", "filename_in_repo": "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", "save_filename": "Wan2.1_14b_Text_to_Video_FP8.safetensors"},
                    {"name": "Wan 2.1 I2V 14B 480p FP16", "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename_in_repo": "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", "save_filename": "Wan2.1_14b_Image_to_Video_480p.safetensors"},
                    {"name": "Wan 2.1 I2V 14B 480p FP8", "repo_id": "Kijai/WanVideo_comfy", "filename_in_repo": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", "save_filename": "Wan2.1_14b_Image_to_Video_480p_FP8.safetensors"},
                    {"name": "Wan 2.1 I2V 14B 720p FP16", "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename_in_repo": "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", "save_filename": "Wan2.1_14b_Image_to_Video_720p.safetensors"},
                    {"name": "Wan 2.1 I2V 14B 720p FP8", "repo_id": "Kijai/WanVideo_comfy", "filename_in_repo": "Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors", "save_filename": "Wan2.1_14b_Image_to_Video_720p_FP8.safetensors"},
                    {"name": "WanVideo 2.1 MultiTalk 14B FP32", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "WanVideo_2_1_Multitalk_14B_fp32.safetensors", "save_filename": "WanVideo_2_1_Multitalk_14B_fp32.safetensors"},
                    {"name": "Wan 2.1 T2V 14B 720p GGUF Q8", "repo_id": "city96/Wan2.1-T2V-14B-gguf", "filename_in_repo": "wan2.1-t2v-14b-Q8_0.gguf", "save_filename": "Wan2.1_14b_Text_to_Video_GGUF_Q8.gguf"},
                    {"name": "Wan 2.1 T2V 14B 720p GGUF Q6_K", "repo_id": "city96/Wan2.1-T2V-14B-gguf", "filename_in_repo": "wan2.1-t2v-14b-Q6_K.gguf", "save_filename": "Wan2.1_14b_Text_to_Video_GGUF_Q6_K.gguf"},
                    {"name": "Wan 2.1 T2V 14B 720p GGUF Q5_K_M", "repo_id": "city96/Wan2.1-T2V-14B-gguf", "filename_in_repo": "wan2.1-t2v-14b-Q5_K_M.gguf", "save_filename": "Wan2.1_14b_Text_to_Video_GGUF_Q5_K_M.gguf"},
                    {"name": "Wan 2.1 T2V 14B 720p GGUF Q4_K_M", "repo_id": "city96/Wan2.1-T2V-14B-gguf", "filename_in_repo": "wan2.1-t2v-14b-Q4_K_M.gguf", "save_filename": "Wan2.1_14b_Text_to_Video_GGUF_Q4_K_M.gguf"},
                    {"name": "Wan 2.1 I2V 14B 480p GGUF Q8", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan21_i2v_480p_14B_Q8.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_480p_GGUF_Q8.gguf"},
                    {"name": "Wan 2.1 I2V 14B 480p GGUF Q6_K", "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf", "filename_in_repo": "wan2.1-i2v-14b-480p-Q6_K.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_480p_GGUF_Q6_K.gguf"},
                    {"name": "Wan 2.1 I2V 14B 480p GGUF Q5_K_M", "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf", "filename_in_repo": "wan2.1-i2v-14b-480p-Q5_K_M.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_480p_GGUF_Q5_K_M.gguf"},
                    {"name": "Wan 2.1 I2V 14B 480p GGUF Q4_K_M", "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf", "filename_in_repo": "wan2.1-i2v-14b-480p-Q4_K_M.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_480p_GGUF_Q4_K_M.gguf"},
                    {"name": "Wan 2.1 I2V 14B 720p GGUF Q8", "repo_id": "city96/Wan2.1-I2V-14B-720P-gguf", "filename_in_repo": "wan2.1-i2v-14b-720p-Q8_0.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_720p_GGUF_Q8.gguf"},
                    {"name": "Wan 2.1 I2V 14B 720p GGUF Q6_K", "repo_id": "city96/Wan2.1-I2V-14B-720P-gguf", "filename_in_repo": "wan2.1-i2v-14b-720p-Q6_K.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_720p_GGUF_Q6_K.gguf"},
                    {"name": "Wan 2.1 I2V 14B 720p GGUF Q5_K_M", "repo_id": "city96/Wan2.1-I2V-14B-720P-gguf", "filename_in_repo": "wan2.1-i2v-14b-720p-Q5_K_M.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_720p_GGUF_Q5_K_M.gguf"},
                    {"name": "Wan 2.1 I2V 14B 720p GGUF Q4_K_M", "repo_id": "city96/Wan2.1-I2V-14B-720P-gguf", "filename_in_repo": "wan2.1-i2v-14b-720p-Q4_K_M.gguf", "save_filename": "Wan2.1_14b_Image_to_Video_720p_GGUF_Q4_K_M.gguf"},
                 ]
             },
            "Wan 2.1 FusionX Models": {
                 "info": ("Wan 2.1 FusionX text-to-video and image-to-video models with enhanced performance. GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).\n\n"
                          "**Extremely Important How To Use Parameters and Guide:** [Wan 2.1 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21-parameters)"),
                 "target_dir_key": "diffusion_models",
                 "models": [
                    {"name": "Wan 2.1 FusionX T2V 14B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2_1_14B_FusionX_T2V_fp16.safetensors", "save_filename": "Wan2.1_14b_FusionX_Text_to_Video_FP16.safetensors"},
                    {"name": "Wan 2.1 FusionX T2V 14B FP8", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2_1_14B_FusionX_T2V_fp8.safetensors", "save_filename": "Wan2.1_14b_FusionX_Text_to_Video_FP8.safetensors"},
                    {"name": "Wan 2.1 FusionX I2V 14B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2_1_14B_FusionX_I2V_fp16.safetensors", "save_filename": "Wan2.1_14b_FusionX_Image_to_Video_FP16.safetensors"},
                    {"name": "Wan 2.1 FusionX I2V 14B FP8", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2_1_14B_FusionX_I2V_fp8.safetensors", "save_filename": "Wan2.1_14b_FusionX_Image_to_Video_FP8.safetensors"},
                    wan_fusionx_t2v_gguf_q8_entry,
                    wan_fusionx_t2v_gguf_q6_entry,
                    wan_fusionx_t2v_gguf_q5_entry,
                    wan_fusionx_t2v_gguf_q4_entry,
                    wan_fusionx_i2v_gguf_q8_entry,
                    wan_fusionx_i2v_gguf_q6_entry,
                    wan_fusionx_i2v_gguf_q5_entry,
                    wan_fusionx_i2v_gguf_q4_entry,
                 ]
             },
            "Wan 2.1 LoRAs": {
                 "info": ("Wan 2.1 LoRA models for enhanced performance and specialized use cases. See SwarmUI Video Docs for usage details on CFG, Steps, FPS, and Trim Video Start Frames.\n\n"
                          "**Extremely Important How To Use Parameters and Guide:** [Wan 2.1 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21-parameters)"),
                 "target_dir_key": "Lora",
                 "models": [
                    wan_causvid_14b_lora_v2_entry,
                    wan_causvid_14b_lora_entry,
                    wan_causvid_1_3b_lora_entry,
                    wan_self_forcing_lora_entry,
                    {"name": "Phantom Wan 14B FusionX LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Phantom_Wan_14B_FusionX_LoRA.safetensors", "save_filename": "Phantom_Wan_14B_FusionX_LoRA.safetensors", "target_dir_key": "Lora", "info": "Phantom FusionX LoRA for Wan 2.1 14B models. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan 2.1 I2V 14B FusionX LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2.1_Image_to_Video_14B_FusionX_LoRA.safetensors", "save_filename": "Wan2.1_Image_to_Video_14B_FusionX_LoRA.safetensors", "target_dir_key": "Lora", "info": "FusionX I2V LoRA for Wan 2.1 14B models. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan 2.1 T2V 14B FusionX LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors", "save_filename": "Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors", "target_dir_key": "Lora", "info": "FusionX T2V LoRA for Wan 2.1 14B models. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    wan_lightx2v_lora_entry,
                 ]
             },
            "Wan 2.2 Official Models": {
                 "info": ("Official Wan 2.2 text-to-video and image-to-video models. Includes both high and low noise variants.\n\n"
                          "**Extremely Important How To Use Parameters and Guide:** [Wan 2.2 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-22)"),
                 "target_dir_key": "diffusion_models",
                 "models": [
                    {"name": "Wan 2.2 I2V High Noise 14B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_i2v_high_noise_14B_fp16.safetensors", "save_filename": "wan2.2_i2v_high_noise_14B_fp16.safetensors"},
                    {"name": "Wan 2.2 I2V High Noise 14B FP8 Scaled", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "save_filename": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"},
                    {"name": "Wan 2.2 I2V Low Noise 14B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_i2v_low_noise_14B_fp16.safetensors", "save_filename": "wan2.2_i2v_low_noise_14B_fp16.safetensors"},
                    {"name": "Wan 2.2 I2V Low Noise 14B FP8 Scaled", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "save_filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"},
                    {"name": "Wan 2.2 T2V High Noise 14B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_t2v_high_noise_14B_fp16.safetensors", "save_filename": "wan2.2_t2v_high_noise_14B_fp16.safetensors"},
                    {"name": "Wan 2.2 T2V High Noise 14B FP8 Scaled", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "save_filename": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
                    {"name": "Wan 2.2 T2V Low Noise 14B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_t2v_low_noise_14B_fp16.safetensors", "save_filename": "wan2.2_t2v_low_noise_14B_fp16.safetensors"},
                    {"name": "Wan 2.2 T2V Low Noise 14B FP8 Scaled", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "save_filename": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"},
                    {"name": "Wan 2.2 TI2V 5B FP16", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_ti2v_5B_fp16.safetensors", "save_filename": "wan2.2_ti2v_5B_fp16.safetensors"},
                 ]
             },
            "Wan 2.2 LoRAs": {
                 "info": ("Wan 2.2 LoRA models for enhanced performance and specialized use cases. See SwarmUI Video Docs for usage details on CFG, Steps, FPS, and Trim Video Start Frames.\n\n"
                          "**Extremely Important How To Use Parameters and Guide:** [Wan 2.2 Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-22)"),
                 "target_dir_key": "Lora",
                 "models": [
                    {"name": "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_Low", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_Low.safetensors", "save_filename": "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_Low.safetensors", "target_dir_key": "Lora", "info": "Wan 2.2 T2V LoRA Low Noise variant for fast 4-step generation. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_High", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_High.safetensors", "save_filename": "Wan2_2-T2V-A14B-4steps-lora-rank64-Seko-V1_1_High.safetensors", "target_dir_key": "Lora", "info": "Wan 2.2 T2V LoRA High Noise variant for fast 4-step generation. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low.safetensors", "save_filename": "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low.safetensors", "target_dir_key": "Lora", "info": "Wan 2.2 I2V LoRA Low Noise variant for fast 4-step generation. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High.safetensors", "save_filename": "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High.safetensors", "target_dir_key": "Lora", "info": "Wan 2.2 I2V LoRA High Noise variant for fast 4-step generation. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                 ]
             },
            "Hunyuan Models": {
                "info": ("Hunyuan text-to-video and image-to-video models. GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).\n\n"
                         "**Extremely Important How To Use Parameters and Guide:** [Hunyuan Video Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#hunyuan-video-parameters)"),
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "HunYuan T2V 720p BF16", "repo_id": "Comfy-Org/HunyuanVideo_repackaged", "filename_in_repo": "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors", "save_filename": "HunYuan_Text_to_Video.safetensors"},
                    {"name": "HunYuan I2V 720p BF16", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_I2V_720_fixed_bf16.safetensors", "save_filename": "HunYuan_Image_to_Video.safetensors"},
                    {"name": "HunYuan T2V 720p CFG Distill BF16", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_720_cfgdistill_bf16.safetensors", "save_filename": "HunYuan_Text_to_Video_CFG_Distill.safetensors"},
                    {"name": "HunYuan T2V 720p CFG Distill FP8 Scaled", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors", "save_filename": "HunYuan_Text_to_Video_CFG_Distill_FP8_Scaled.safetensors"},
                    {"name": "HunYuan I2V 720p FP8 Scaled", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_I2V_720_fixed_fp8_e4m3fn.safetensors", "save_filename": "HunYuan_Image_to_Video_FP8_Scaled.safetensors"},
                    {"name": "HunYuan I2V 720p GGUF Q8", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_I2V-Q8_0.gguf", "save_filename": "HunYuan_Image_to_Video_GGUF_Q8.gguf"},
                    {"name": "HunYuan I2V 720p GGUF Q6_K", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_I2V-Q6_K.gguf", "save_filename": "HunYuan_Image_to_Video_GGUF_Q6_K.gguf"},
                    {"name": "HunYuan I2V 720p GGUF Q4_K_S", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_I2V-Q4_K_S.gguf", "save_filename": "HunYuan_Image_to_Video_GGUF_Q4_K_S.gguf"},
                ]
            },
            "Fast Hunyuan Models - 6 Steps": {
                "info": ("Faster distilled Hunyuan text-to-video models (6 steps). GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).\n\n"
                         "**Extremely Important How To Use Parameters and Guide:** [FastVideo Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#fastvideo)"),
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "FAST HunYuan T2V 720p GGUF BF16", "repo_id": "city96/FastHunyuan-gguf", "filename_in_repo": "fast-hunyuan-video-t2v-720p-BF16.gguf", "save_filename": "FAST_HunYuan_Text_to_Video_GGUF_BF16.gguf"},
                    {"name": "FAST HunYuan T2V 720p FP8", "repo_id": "Kijai/HunyuanVideo_comfy", "filename_in_repo": "hunyuan_video_FastVideo_720_fp8_e4m3fn.safetensors", "save_filename": "FAST_HunYuan_Text_to_Video_FP8.safetensors"},
                    {"name": "FAST HunYuan T2V 720p GGUF Q8", "repo_id": "city96/FastHunyuan-gguf", "filename_in_repo": "fast-hunyuan-video-t2v-720p-Q8_0.gguf", "save_filename": "FAST_HunYuan_Text_to_Video_GGUF_Q8.gguf"},
                    {"name": "FAST HunYuan T2V 720p GGUF Q6_K", "repo_id": "city96/FastHunyuan-gguf", "filename_in_repo": "fast-hunyuan-video-t2v-720p-Q6_K.gguf", "save_filename": "FAST_HunYuan_Text_to_Video_GGUF_Q6_K.gguf"},
                    {"name": "FAST HunYuan T2V 720p GGUF Q5_K_M", "repo_id": "city96/FastHunyuan-gguf", "filename_in_repo": "fast-hunyuan-video-t2v-720p-Q5_K_M.gguf", "save_filename": "FAST_HunYuan_Text_to_Video_GGUF_Q5_K_M.gguf"},
                    {"name": "FAST HunYuan T2V 720p GGUF Q4_K_M", "repo_id": "city96/FastHunyuan-gguf", "filename_in_repo": "fast-hunyuan-video-t2v-720p-Q4_K_M.gguf", "save_filename": "FAST_HunYuan_Text_to_Video_GGUF_Q4_K_M.gguf"},
                ]
            },
             "SkyReels HunYuan Models": {
                "info": ("SkyReels fine-tuned Hunyuan models. GGUF Quality: Q8 > Q6 > Q5 > Q4 (K_M > K_S > K > 1 > 0).\n\n"
                         "**Extremely Important How To Use Parameters and Guide:** [SkyReels Text2Video Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#skyreels-text2video)"),
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "SkyReels HunYuan T2V 720p BF16", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels_hunyuan_t2v_bf16.safetensors", "save_filename": "SkyReels_Text_to_Video.safetensors"},
                    {"name": "SkyReels HunYuan I2V 720p BF16", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels_hunyuan_i2v_bf16.safetensors", "save_filename": "SkyReels_Image_to_Video.safetensors"},
                    {"name": "SkyReels HunYuan T2V 720p FP8", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels_hunyuan_t2v_fp8_e4m3fn.safetensors", "save_filename": "SkyReels_Text_to_Video_FP8.safetensors"},
                    {"name": "SkyReels HunYuan I2V 720p FP8", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels_hunyuan_i2v_fp8_e4m3fn.safetensors", "save_filename": "SkyReels_Image_to_Video_FP8.safetensors"},
                    {"name": "SkyReels HunYuan I2V 720p GGUF Q8", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels-hunyuan-I2V-Q8_0.gguf", "save_filename": "SkyReels_Image_to_Video_GGUF_Q8.gguf"},
                    {"name": "SkyReels HunYuan I2V 720p GGUF Q6_K", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels-hunyuan-I2V-Q6_K.gguf", "save_filename": "SkyReels_Image_to_Video_GGUF_Q6_K.gguf"},
                    {"name": "SkyReels HunYuan I2V 720p GGUF Q5_K_M", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels-hunyuan-I2V-Q5_K_M.gguf", "save_filename": "SkyReels_Image_to_Video_GGUF_Q5_K_M.gguf"},
                    {"name": "SkyReels HunYuan I2V 720p GGUF Q4_K_S", "repo_id": "Kijai/SkyReels-V1-Hunyuan_comfy", "filename_in_repo": "skyreels-hunyuan-I2V-Q4_K_S.gguf", "save_filename": "SkyReels_Image_to_Video_GGUF_Q4_K_S.gguf"},
                ]
            },
            "Genmo Mochi 1 Models": {
                "info": ("Preview release of Genmo Mochi 1 text-to-video model.\n\n"
                         "**Extremely Important How To Use Parameters and Guide:** [Genmo Mochi 1 Text2Video Parameters](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#genmo-mochi-1-text2video)"),
                "target_dir_key": "diffusion_models",
                "models": [
                    {"name": "Genmo Mochi 1 Preview T2V BF16", "repo_id": "Comfy-Org/mochi_preview_repackaged", "filename_in_repo": "split_files/diffusion_models/mochi_preview_bf16.safetensors", "save_filename": "Genmo_Mochi_1_Text_to_Video.safetensors"},
                    {"name": "Genmo Mochi 1 Preview T2V FP8 Scaled", "repo_id": "Comfy-Org/mochi_preview_repackaged", "filename_in_repo": "split_files/diffusion_models/mochi_preview_fp8_scaled.safetensors", "save_filename": "Genmo_Mochi_1_Text_to_Video_FP8_Scaled.safetensors"},
                ]
            },
             "Lightricks LTX Video Models - Ultra Fast": {
                 "info": (f"Ultra-fast text-to-video and image-to-video models from Lightricks. "
                          f"The companion 'LTX VAE (BF16)' is listed below and also in the VAEs section; it's recommended for the 13B Dev models. "
                          f"{GGUF_QUALITY_INFO} (for GGUF variants)\n\n"
                          "**Extremely Important How To Use Parameters and Guide:** [LTX Video Installation/Usage](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#ltxv-install)"),
                 "target_dir_key": "diffusion_models", # Default for this sub-category (will be used by GGUFs)
                 "models": [
                    {"name": "LTX 2b T2V+I2V 768x512 v0.9.5", "repo_id": "Lightricks/LTX-Video", "filename_in_repo": "ltx-video-2b-v0.9.5.safetensors", "save_filename": "LTX_2b_V_0_9_5.safetensors", "target_dir_key": "Stable-Diffusion"},
                    {"name": "LTX 13B Dev T2V+I2V 0.9.7 (FP16/BF16)", "repo_id": "Lightricks/LTX-Video", "filename_in_repo": "ltxv-13b-0.9.7-dev.safetensors", "save_filename": "LTX_13B_Dev_V_0_9_7.safetensors", "target_dir_key": "Stable-Diffusion"},
                    {"name": "LTX 13B Dev T2V+I2V 0.9.7 FP8", "repo_id": "Lightricks/LTX-Video", "filename_in_repo": "ltxv-13b-0.9.7-dev-fp8.safetensors", "save_filename": "LTX_13B_Dev_V_0_9_7_FP8.safetensors", "target_dir_key": "Stable-Diffusion"},
                    # GGUF models will use the sub-category's default target_dir_key: "diffusion_models"
                    {"name": "LTX 13B Dev T2V+I2V 0.9.7 GGUF Q8_0", "repo_id": "wsbagnsv1/ltxv-13b-0.9.7-dev-GGUF", "filename_in_repo": "ltxv-13b-0.9.7-dev-Q8_0.gguf", "save_filename": "LTX_13B_Dev_V_0_9_7_GGUF_Q8_0.gguf"},
                    {"name": "LTX 13B Dev T2V+I2V 0.9.7 GGUF Q6_K", "repo_id": "wsbagnsv1/ltxv-13b-0.9.7-dev-GGUF", "filename_in_repo": "ltxv-13b-0.9.7-dev-Q6_K.gguf", "save_filename": "LTX_13B_Dev_V_0_9_7_GGUF_Q6_K.gguf"},
                    {"name": "LTX 13B Dev T2V+I2V 0.9.7 GGUF Q5_K_M", "repo_id": "wsbagnsv1/ltxv-13b-0.9.7-dev-GGUF", "filename_in_repo": "ltxv-13b-0.9.7-dev-Q5_K_M.gguf", "save_filename": "LTX_13B_Dev_V_0_9_7_GGUF_Q5_K_M.gguf"},
                    {"name": "LTX 13B Dev T2V+I2V 0.9.7 GGUF Q4_K_M", "repo_id": "wsbagnsv1/ltxv-13b-0.9.7-dev-GGUF", "filename_in_repo": "ltxv-13b-0.9.7-dev-Q4_K_M.gguf", "save_filename": "LTX_13B_Dev_V_0_9_7_GGUF_Q4_K_M.gguf"},
                    ltx_vae_companion_entry, # This entry has its own target_dir_key: "vae"
                 ]
             },
        }
    },
    "LoRA Models": {
        "info": "Readme for Wan 2.1 CausVid LoRA to Speed Up : [LoRA Models](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-causvid---high-speed-14b)",
        "sub_categories": {
            "Wan 2.1 LoRAs": {
                "info": "Wan 2.1 LoRA models for enhanced performance and specialized use cases. See SwarmUI Video Docs for usage details on CFG, Steps, FPS, and Trim Video Start Frames.",
                "target_dir_key": "Lora",
                "models": [
                    wan_causvid_14b_lora_v2_entry,
                    wan_causvid_14b_lora_entry,
                    wan_causvid_1_3b_lora_entry,
                    {"name": "Phantom Wan 14B FusionX LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Phantom_Wan_14B_FusionX_LoRA.safetensors", "save_filename": "Phantom_Wan_14B_FusionX_LoRA.safetensors", "target_dir_key": "Lora", "info": "Phantom FusionX LoRA for Wan 2.1 14B models. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan 2.1 I2V 14B FusionX LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2.1_Image_to_Video_14B_FusionX_LoRA.safetensors", "save_filename": "Wan2.1_Image_to_Video_14B_FusionX_LoRA.safetensors", "target_dir_key": "Lora", "info": "FusionX I2V LoRA for Wan 2.1 14B models. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    {"name": "Wan 2.1 T2V 14B FusionX LoRA", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors", "save_filename": "Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors", "target_dir_key": "Lora", "info": "FusionX T2V LoRA for Wan 2.1 14B models. Saves to Lora folder. See SwarmUI Video Docs for usage details."},
                    wan_self_forcing_lora_entry,
                    wan_lightx2v_lora_entry,
                ]
            },
            "Various LoRAs": {
                "info": "A collection of LoRA models.",
                "target_dir_key": "Lora",
                "models": [
                    {"name": "Migration LoRA Cloth (TTPlanet)", "repo_id": "TTPlanet/Migration_Lora_flux", "filename_in_repo": "Migration_Lora_cloth.safetensors", "save_filename": "Migration_Lora_cloth.safetensors"},
                    {"name": "Figures TTP Migration LoRA (TTPlanet)", "repo_id": "TTPlanet/Migration_Lora_flux", "filename_in_repo": "figures_TTP_Migration.safetensors", "save_filename": "figures_TTP_Migration.safetensors"},
                ]
            }
        }
    },
    "ControlNet Models": {
        "info": "ControlNets",
        "sub_categories": {
            "Various ControlNets": {
                "info": "A collection of ControlNet models.",
                "target_dir_key": "controlnet",
                "models": [
                    wan_uni3c_controlnet_lora_entry,                    
                ]
            }
        }
    },
    "LLM Models": {
        "info": "Large Language Models (LLMs) used for various purposes, such as advanced text encoders or other functionalities.",
        "sub_categories": {
            "General LLMs": {
                "info": "Full LLM model repositories.",
                "target_dir_key": "LLM", # General target for this sub_category
                "models": [
                    {"name": "Meta-Llama-3.1-8B-Instruct (Full Repo)", "repo_id": "unsloth/Meta-Llama-3.1-8B-Instruct", "is_snapshot": True, "target_dir_key": "LLM_unsloth_llama"}
                ]
            }
        }
    },
    "VAE Models": {
        "info": "Variational Autoencoder models, used to improve image quality and details.",
        "sub_categories": {
            "Most Common VAEs (e.g. FLUX and HiDream-I1)": {
                "info": "VAEs commonly used with various models like FLUX and HiDream.",
                "target_dir_key": "vae", # Correct target directory
                "models": [
                    {"name": "FLUX VAE as FLUX_VAE.safetensors (Used by FLUX, HiDream, etc.)", "repo_id": "MonsterMMORPG/Kohya_Train", "filename_in_repo": "ae.safetensors", "save_filename": "FLUX_VAE.safetensors"},
                    {"name": "FLUX VAE as ae.safetensors", "repo_id": "MonsterMMORPG/Kohya_Train", "filename_in_repo": "ae.safetensors", "save_filename": "ae.safetensors"},
                    ltx_vae_companion_entry, # Added the VAE companion here
                    wan_vae_entry,
                    {"name": "Wan 2.2 VAE", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "wan2.2_vae.safetensors", "save_filename": "wan2.2_vae.safetensors", "target_dir_key": "vae"},
                    {"name": "qwen_image_vae.safetensors", "repo_id": "MonsterMMORPG/Wan_GGUF", "filename_in_repo": "qwen_image_vae.safetensors", "save_filename": "qwen_image_vae.safetensors"},
                ]
            },
        }
    },
    "Clip Vision Models": {
        "info": "Vision encoder models, e.g., for image understanding or as part of larger multi-modal systems.",
        "sub_categories": {
            "Standard Clip Vision Models": {
                "info": "Standard CLIP vision encoders used by various models including Wan 2.1.",
                "target_dir_key": "clip_vision",
                "models": [
                    {
                        "name": "CLIP Vision H (Used by Wan 2.1)",
                        "repo_id": "MonsterMMORPG/Wan_GGUF",
                        "filename_in_repo": "clip_vision_h.safetensors",
                        "save_filename": "clip_vision_h.safetensors"
                    }
                ]
            },
            "SigLIP Vision Models": {
                "info": "Sigmoid-Loss for Language-Image Pre-Training (SigLIP) vision encoders. These are typically used by specific model architectures that require them.",
                "target_dir_key": "clip_vision",
                "models": [
                    {
                        "name": "SigLIP Vision Patch14 384px",
                        "repo_id": "Comfy-Org/sigclip_vision_384",
                        "filename_in_repo": "sigclip_vision_patch14_384.safetensors",
                        "save_filename": "sigclip_vision_patch14_384.safetensors"
                    },
                    {
                        "name": "SigLIP SO400M Patch14 384px (Full Repo)",
                        "repo_id": "google/siglip-so400m-patch14-384",
                        "is_snapshot": True,
                        "target_dir_key": "clip_vision_google_siglip"
                    }
                ]
            }
        }
    },
    "ComfyUI Workflows": {
        "info": "Downloadable ComfyUI workflow JSON files or related assets.",
        "sub_categories": {
            "Captioning Workflows": {
                "info": "Workflows and assets related to image captioning.",
                "target_dir_key": "Joy_caption", # General target for this sub_category
                "models": [
                    {"name": "Joy Caption Alpha Two (Full Repo)", "repo_id": "MonsterMMORPG/joy-caption-alpha-two", "is_snapshot": True, "target_dir_key": "Joy_caption_monster_joy"}
                ]
            }
        }
    },

}


LAST_SETTINGS_FILE = "last_settings.json"
MODEL_SIZES_FILE = "model_sizes.json"

# Global variable to store size data
size_data = None

def save_last_settings(path, comfy_ui_structure, forge_structure=False, lowercase_folders=False):
    """Saves the given path, ComfyUI structure, Forge structure, and lowercase folders setting to a JSON file for next startup."""
    try:
        settings = {
            "path": path,
            "comfy_ui_structure": comfy_ui_structure,
            "forge_structure": forge_structure,
            "lowercase_folders": lowercase_folders
        }
        with open(LAST_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        add_log(f"Saved settings - Path: '{path}', ComfyUI structure: {comfy_ui_structure}, Forge structure: {forge_structure}, Lowercase folders: {lowercase_folders}")
        return f"✓ Settings saved: {path} (ComfyUI: {comfy_ui_structure}, Forge: {forge_structure}, Lowercase: {lowercase_folders})"
    except Exception as e:
        error_msg = f"ERROR: Could not save settings to file: {e}"
        add_log(error_msg)
        return f"✗ Error saving settings: {e}"

def load_last_settings():
    """Loads the last used settings from the JSON file if it exists. Also handles backward compatibility with old text file format."""
    try:
        # First try to load from new JSON format
        if os.path.exists(LAST_SETTINGS_FILE):
            with open(LAST_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            path = settings.get("path", "")
            comfy_ui_structure = settings.get("comfy_ui_structure", False)
            forge_structure = settings.get("forge_structure", False)
            lowercase_folders = settings.get("lowercase_folders", False)
            
            if path and os.path.isdir(path):
                print(f"Loaded saved settings from {LAST_SETTINGS_FILE}: Path='{path}', ComfyUI={comfy_ui_structure}, Forge={forge_structure}, Lowercase={lowercase_folders}")
                return path, comfy_ui_structure, forge_structure, lowercase_folders
            else:
                print(f"Saved path '{path}' from {LAST_SETTINGS_FILE} is not a valid directory. Using defaults.")
                return None, False, False, False
        
        # Backward compatibility: check for old text file format
        old_file = "last_model_path.txt"
        if os.path.exists(old_file):
            with open(old_file, 'r', encoding='utf-8') as f:
                path = f.read().strip()
            if path and os.path.isdir(path):
                print(f"Loaded saved path from legacy file {old_file}: {path} (migrating to new format)")
                # Save to new format and remove old file
                try:
                    save_last_settings(path, False, False, False)
                    os.remove(old_file)
                    print(f"Migrated settings to new format and removed legacy file.")
                except Exception as e:
                    print(f"Warning: Could not migrate legacy settings: {e}")
                return path, False, False, False
            else:
                print(f"Legacy saved path '{path}' from {old_file} is not a valid directory. Using defaults.")
                return None, False, False, False
        
        print(f"No saved settings file ({LAST_SETTINGS_FILE}) found. Using defaults.")
        return None, False, False, False
    except Exception as e:
        print(f"ERROR: Could not load saved settings from {LAST_SETTINGS_FILE}: {e}. Using defaults.")
        return None, False, False, False

def load_last_path():
    """Legacy function for backward compatibility - loads only the path."""
    path, _, _, _ = load_last_settings()
    return path

def load_model_sizes():
    """Load model size data from JSON file."""
    global size_data
    try:
        if os.path.exists(MODEL_SIZES_FILE):
            with open(MODEL_SIZES_FILE, 'r', encoding='utf-8') as f:
                size_data = json.load(f)
            fetch_date = size_data.get("fetch_date", "Unknown")
            model_count = len(size_data.get("models", {}))
            bundle_count = len(size_data.get("bundles", {}))
            print(f"Loaded model size data from {MODEL_SIZES_FILE} (fetched: {fetch_date})")
            print(f"  - {model_count} individual models with size data")
            print(f"  - {bundle_count} bundles with size data")
            
            # Debug: Show first few model keys for troubleshooting
            if model_count > 0:
                model_keys = list(size_data.get("models", {}).keys())
                print(f"  - Sample model keys: {model_keys[:3]}...")
                
                # Check for models with errors
                error_count = 0
                success_count = 0
                for key, model_info in size_data.get("models", {}).items():
                    if model_info.get("error"):
                        error_count += 1
                    elif model_info.get("size_gb", 0) > 0:
                        success_count += 1
                        
                print(f"  - Successfully fetched sizes: {success_count}")
                print(f"  - Models with errors: {error_count}")
                
                # Show some examples of models with sizes
                models_with_sizes = []
                for key, model_info in size_data.get("models", {}).items():
                    if model_info.get("size_gb", 0) > 0:
                        models_with_sizes.append(f"{model_info['name']} ({model_info['size_gb']:.2f} GB)")
                        if len(models_with_sizes) >= 3:
                            break
                if models_with_sizes:
                    print(f"  - Example models with sizes: {', '.join(models_with_sizes)}")
                
            return True
        else:
            print(f"No model size data file ({MODEL_SIZES_FILE}) found. Sizes will not be displayed.")
            print(f"To generate size data, run: python fetch_model_sizes.py")
            return False
    except Exception as e:
        print(f"ERROR: Could not load model size data from {MODEL_SIZES_FILE}: {e}")
        print(f"Traceback: {str(e)}")
        return False

def get_subcategory_total_size_display(cat_name, sub_cat_name, models_list):
    """Get total size display string for all models in a subcategory."""
    if not size_data or not size_data.get("models"):
        return f" ({len(models_list)} models - sizes unknown)"
    
    total_size_gb = 0.0
    models_with_size = 0
    models_with_error = 0
    
    for model_info in models_list:
        model_name = model_info.get("name", "Unknown")
        model_key = f"{cat_name}::{sub_cat_name}::{model_name}"
        model_size_info = size_data.get("models", {}).get(model_key)
        
        # If exact match not found, try fuzzy matching like in get_model_size_display
        if not model_size_info:
            possible_matches = []
            for key, info in size_data.get("models", {}).items():
                if model_name.lower() in key.lower() or info.get("name", "").lower() == model_name.lower():
                    possible_matches.append((key, info))
            
            if len(possible_matches) == 1:
                model_size_info = possible_matches[0][1]
            elif len(possible_matches) > 1:
                for key, info in possible_matches:
                    if key.endswith(f"::{model_name}"):
                        model_size_info = info
                        break
        
        if model_size_info:
            size_gb = model_size_info.get("size_gb", 0.0)
            if size_gb > 0:
                total_size_gb += size_gb
                models_with_size += 1
            elif model_size_info.get("error"):
                models_with_error += 1
    
    total_models = len(models_list)
    if models_with_size == 0:
        return f" ({total_models} models - total size unknown)"
    elif models_with_size == total_models:
        return f" ({total_models} models - Total: {total_size_gb:.2f} GB)"
    else:
        missing_count = total_models - models_with_size - models_with_error
        return f" ({total_models} models - {total_size_gb:.2f} GB + {missing_count} unknown)"

def get_model_size_display(cat_name, sub_cat_name, model_name):
    """Get size display string for a model."""
    if not size_data or not size_data.get("models"):
        return " (Size unknown - run fetch_model_sizes.py)"
    
    model_key = f"{cat_name}::{sub_cat_name}::{model_name}"
    model_info = size_data.get("models", {}).get(model_key)
    
    # If exact match not found, try fuzzy matching
    if not model_info:
        # Try to find the model by checking all keys that contain the model name
        possible_matches = []
        for key, info in size_data.get("models", {}).items():
            # Check if the model name is similar (allowing for small variations)
            if model_name.lower() in key.lower() or info.get("name", "").lower() == model_name.lower():
                possible_matches.append((key, info))
        
        # If we found exactly one match, use it
        if len(possible_matches) == 1:
            model_key, model_info = possible_matches[0]
            print(f"DEBUG: Used fuzzy match for '{model_name}': '{model_key}'")
        elif len(possible_matches) > 1:
            # Multiple matches, try to find the best one
            for key, info in possible_matches:
                if key.endswith(f"::{model_name}"):  # Exact model name match at the end
                    model_key, model_info = key, info
                    print(f"DEBUG: Used exact model name match for '{model_name}': '{model_key}'")
                    break
    
    # Debug: Print first few lookups to help troubleshoot
    if not hasattr(get_model_size_display, '_debug_count'):
        get_model_size_display._debug_count = 0
    
    if get_model_size_display._debug_count < 5:
        get_model_size_display._debug_count += 1
        print(f"DEBUG: Model key lookup #{get_model_size_display._debug_count}: '{model_key}' -> {'Found' if model_info else 'Not found'}")
        
        # If still not found, show available keys for this model name
        if not model_info:
            similar_keys = []
            for key in size_data.get("models", {}).keys():
                if any(part.lower() in key.lower() for part in model_name.split()):
                    similar_keys.append(key)
            if similar_keys:
                print(f"DEBUG: Keys containing parts of '{model_name}': {similar_keys[:3]}")
    
    if model_info:
        size_gb = model_info.get("size_gb", 0.0)
        if size_gb > 0:
            return f" ({size_gb:.2f} GB)"
        elif model_info.get("error"):
            error_msg = model_info.get("error", "Unknown error")
            return f" (Size error: {error_msg})"
        else:
            return " (Size: 0 GB)"
    else:
        # Debug: Print missing model key for troubleshooting
        print(f"DEBUG: No size data found for model key: {model_key}")
        return " (Size not found)"

def get_bundle_size_display(cat_name, bundle_index):
    """Get size display string for a bundle."""
    if not size_data or not size_data.get("bundles"):
        return " (Bundle size unknown - run fetch_model_sizes.py)"
    
    bundle_key = f"{cat_name}::bundle_{bundle_index}"
    bundle_info = size_data.get("bundles", {}).get(bundle_key)
    
    if bundle_info:
        total_size_gb = bundle_info.get("total_size_gb", 0.0)
        model_count = bundle_info.get("model_count", 0)
        if total_size_gb > 0:
            return f" (Total: {total_size_gb:.2f} GB, {model_count} models)"
        else:
            return f" (Bundle size: 0 GB, {model_count} models)"
    else:
        # Debug: Print missing bundle key for troubleshooting
        print(f"DEBUG: No size data found for bundle key: {bundle_key}")
        # List available bundle keys for debugging
        available_bundle_keys = list(size_data.get("bundles", {}).keys())
        if available_bundle_keys:
            print(f"DEBUG: Available bundle keys: {available_bundle_keys[:3]}")
        return " (Bundle size not found)"

def get_bundle_with_sizes_info(cat_name, bundle_index, original_info):
    """Get bundle info with sizes integrated into the Includes section."""
    if not size_data or not size_data.get("bundles"):
        return original_info + "\n\n*Note: Bundle size information unavailable. Run `python fetch_model_sizes.py` to generate size data.*"
    
    bundle_key = f"{cat_name}::bundle_{bundle_index}"
    bundle_info = size_data.get("bundles", {}).get(bundle_key)
    
    if not bundle_info:
        return original_info + "\n\n*Note: Bundle size information not found.*"
    
    models = bundle_info.get("models", [])
    if not models:
        return original_info + "\n\n*Note: No model size data found for this bundle.*"
    
    # Create a mapping of model names to their sizes
    model_size_map = {}
    for model in models:
        name = model.get("name", "Unknown")
        size_gb = model.get("size_gb", 0.0)
        error = model.get("error")
        
        if error:
            model_size_map[name] = f"Error: {error}"
        elif size_gb > 0:
            model_size_map[name] = f"{size_gb:.2f} GB"
        else:
            model_size_map[name] = "0 GB"
    
    # Process the original info to add sizes to the includes section
    updated_info = original_info
    
    # Split the info into lines and process each line in the includes section
    lines = updated_info.split('\n')
    in_includes_section = False
    processed_lines = []
    
    for line in lines:
        if "**Includes:**" in line:
            in_includes_section = True
            processed_lines.append(line)
        elif in_includes_section and line.strip().startswith("- "):
            # Extract the model name from the line (remove the "- " prefix)
            model_name_in_line = line.strip()[2:]  # Remove "- "
            
            # Try to find a matching model size using fuzzy matching
            matched_size = None
            
            # First try exact matches
            for model_name, size_display in model_size_map.items():
                if model_name == model_name_in_line or model_name in model_name_in_line or model_name_in_line in model_name:
                    matched_size = size_display
                    break
            
            # If no exact match, try fuzzy matching
            if not matched_size:
                for model_name, size_display in model_size_map.items():
                    # Check if key parts of the model name match
                    model_parts = model_name.lower().split()
                    line_parts = model_name_in_line.lower().split()
                    
                    # If at least 2 significant words match, consider it a match
                    significant_matches = 0
                    for part in model_parts:
                        if len(part) > 3:  # Only consider words longer than 3 characters
                            if any(part in line_part for line_part in line_parts):
                                significant_matches += 1
                    
                    if significant_matches >= 2:
                        matched_size = size_display
                        break
            
            if matched_size:
                processed_lines.append(f"{line} ({matched_size})")
            else:
                processed_lines.append(f"{line} (Size unknown)")
        elif in_includes_section and line.strip() and not line.strip().startswith("- ") and "**" not in line:
            # End of includes section
            in_includes_section = False
            processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def get_default_base_path():
    """Determines the default base path based on the OS and known paths."""
    # First check for saved path
    saved_path = load_last_path()
    if saved_path:
        return saved_path
    
    # If no saved path, use original logic
    system = platform.system()
    if system == "Windows":
        swarm_path = os.environ.get("SWARM_MODEL_PATH")
        if swarm_path and os.path.isdir(swarm_path): return swarm_path
        return os.path.join(os.getcwd(), "SwarmUI", "Models")
    else:  # Linux/Unix systems
        swarm_path = os.environ.get("SWARM_MODEL_PATH")
        if swarm_path and os.path.isdir(swarm_path): return swarm_path
        if os.path.exists("/home/Ubuntu/apps/StableSwarmUI"):
            return "/home/Ubuntu/apps/StableSwarmUI/Models"
        elif os.path.exists("/workspace/SwarmUI"):
            return "/workspace/SwarmUI/Models"
        else:
            return os.path.join(os.getcwd(), "SwarmUI", "Models")

def get_default_comfy_ui_structure():
    """Determines the default ComfyUI structure setting from saved settings."""
    _, comfy_ui_structure, _, _ = load_last_settings()
    return comfy_ui_structure

def get_default_forge_structure():
    """Determines the default Forge structure setting from saved settings."""
    _, _, forge_structure, _ = load_last_settings()
    return forge_structure

def get_default_lowercase_folders():
    """Determines the default lowercase folders setting from saved settings."""
    _, _, _, lowercase_folders = load_last_settings()
    return lowercase_folders

DEFAULT_BASE_PATH = get_default_base_path()

BASE_SUBDIRS = { # Renamed from SUBDIRS
    "vae": "vae",
    "diffusion_models": "diffusion_models",
    "Stable-Diffusion": "Stable-Diffusion",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "yolov8": "yolov8",
    "style_models": "style_models",
    "Lora": "Lora", # Default Lora, will be changed if ComfyUI mode is on
    "upscale_models": "upscale_models",
    "LLM": "LLM",
    "Joy_caption": "Joy_caption",
    "clip_vision_google_siglip": "clip_vision/google--siglip-so400m-patch14-384",
    "LLM_unsloth_llama": "LLM/unsloth--Meta-Llama-3.1-8B-Instruct",
    "Joy_caption_monster_joy": "Joy_caption/cgrkzexw-599808",
    "controlnet": "controlnet",
}

def get_current_subdirs(is_comfy_ui_structure: bool, is_forge_structure: bool = False):
    """Returns the subdirectory mapping based on ComfyUI or Forge structure flags."""
    current_s = BASE_SUBDIRS.copy()
    if is_comfy_ui_structure:
        current_s["Lora"] = "loras" # Change Lora to loras for ComfyUI
    elif is_forge_structure:
        # Forge WebUI folder structure based on MODEL_SUPPORT_README.md from sd-webui-forge-classic
        # Reference: E:\Forge_Neo_v1\sd-webui-forge-classic\MODEL_SUPPORT_README.md
        
        # Main checkpoint/diffusion models go to Stable-diffusion folder
        current_s["Stable-diffusion"] = "Stable-diffusion"  # Main checkpoints (lowercase d)
        current_s["Stable-Diffusion"] = "Stable-diffusion"  # Handle both cases for compatibility
        current_s["diffusion_models"] = "Stable-diffusion"  # Map ALL diffusion_models to Stable-diffusion
        
        # VAE models - Forge uses "VAE" folder
        current_s["vae"] = "VAE"  # VAE folder in Forge
        current_s["VAE"] = "VAE"  # Also handle uppercase
        
        # LoRA models - Forge uses "Lora" folder
        current_s["Lora"] = "Lora"  # Lora folder in Forge
        current_s["lora"] = "Lora"  # Map lowercase variant
        current_s["loras"] = "Lora"  # Map ComfyUI style to Forge style
        
        # Text encoders - Forge uses text_encoder folder for CLIP/T5 models
        current_s["clip"] = "text_encoder"  # CLIP models go to text_encoder folder
        current_s["text_encoder"] = "text_encoder"  # Text encoder folder
        current_s["clip_vision"] = "text_encoder"  # CLIP vision also goes to text_encoder
        current_s["t5"] = "text_encoder"  # T5 models also go here
        current_s["umt5"] = "text_encoder"  # UMT5 models also go here
        
        # ControlNet models
        current_s["controlnet"] = "ControlNet"  # ControlNet folder
        current_s["ControlNet"] = "ControlNet"  # Handle uppercase variant
        
        # ControlNet Preprocessor models
        current_s["controlnetpreprocessor"] = "ControlNetPreprocessor"
        current_s["ControlNetPreprocessor"] = "ControlNetPreprocessor"
        current_s["preprocessor"] = "ControlNetPreprocessor"
        
        # ALL Upscaler models go to single ESRGAN folder per README (lines 175-184)
        current_s["upscale_models"] = "ESRGAN"  # Default upscalers to ESRGAN
        current_s["ESRGAN"] = "ESRGAN"  # ESRGAN models
        current_s["RealESRGAN"] = "ESRGAN"  # Map to single ESRGAN folder
        current_s["BSRGAN"] = "ESRGAN"  # Map to single ESRGAN folder
        current_s["DAT"] = "ESRGAN"  # Map to single ESRGAN folder
        current_s["SwinIR"] = "ESRGAN"  # Map to single ESRGAN folder
        current_s["ScuNET"] = "ESRGAN"  # Map to single ESRGAN folder
        current_s["upscalers"] = "ESRGAN"  # Any upscaler goes to ESRGAN
        
        # Embeddings folder
        current_s["embeddings"] = "embeddings"  # Textual inversion embeddings
        current_s["embedding"] = "embeddings"  # Alternative naming
        current_s["textual_inversion"] = "embeddings"  # Alternative naming
        
        # Diffusers format models folder
        current_s["diffusers"] = "diffusers"  # Diffusers format models
        current_s["diffusion"] = "diffusers"  # Alternative naming
        
        # Face restoration models (keeping for backward compatibility, though not in official README)
        current_s["Codeformer"] = "Codeformer"  # Codeformer models
        current_s["GFPGAN"] = "GFPGAN"  # GFPGAN models
        
        # Interrogation/captioning models (keeping for backward compatibility)
        current_s["BLIP"] = "BLIP"  # BLIP models
        current_s["deepbooru"] = "deepbooru"  # Deepbooru models (lowercase)
        
        # Additional model types (keeping for backward compatibility)
        current_s["hypernetworks"] = "hypernetworks"  # Hypernetworks (lowercase)
        current_s["LyCORIS"] = "LyCORIS"  # LyCORIS networks
        
        # Note: These folders exist in Forge but may vary by installation
    return current_s

def find_actual_cased_directory_component(parent_dir: str, component_name: str) -> str | None:
    """
    Finds an existing directory component case-insensitively within parent_dir.
    Returns the actual cased name if found as a directory, otherwise None.
    """
    if not os.path.isdir(parent_dir):
        return None
    name_lower = component_name.lower()
    try:
        for item in os.listdir(parent_dir):
            if item.lower() == name_lower:
                if os.path.isdir(os.path.join(parent_dir, item)):
                    return item
    except OSError: # Permission denied, etc.
        pass
    return None

def resolve_target_directory(base_dir: str, relative_path_str: str, lowercase_folders: bool = False) -> str:
    """
    Resolves/constructs a target directory path. On non-Windows systems,
    it attempts to find existing path components case-insensitively.
    The returned path is what should be used for os.makedirs().
    
    Args:
        base_dir: The base directory path
        relative_path_str: The relative path string
        lowercase_folders: If True, convert all directory names to lowercase
    """
    # Normalize relative_path_str once
    normalized_relative_path = os.path.normpath(relative_path_str)
    
    # Apply lowercase to path if requested
    if lowercase_folders:
        normalized_relative_path = normalized_relative_path.lower()

    if platform.system() == "Windows":
        return os.path.join(base_dir, normalized_relative_path)

    # Linux/Mac
    current_path = base_dir
    # Split normalized_relative_path into components
    components = []
    head, tail = os.path.split(normalized_relative_path)
    while tail:
        components.insert(0, tail)
        head, tail = os.path.split(head)
    if head: # If there's a remaining head (e.g. from an absolute path, though not expected here)
        components.insert(0, head)
    
    # Filter out empty or "." components that might result from normpath or splitting
    components = [comp for comp in components if comp and comp != '.']

    for component in components:
        actual_cased_comp = None
        if os.path.isdir(current_path): # Only scan if parent is an existing directory
             actual_cased_comp = find_actual_cased_directory_component(current_path, component)

        if actual_cased_comp:
            current_path = os.path.join(current_path, actual_cased_comp)
        else:
            current_path = os.path.join(current_path, component)
            
    return current_path


def ensure_directories_exist(base_path: str, is_comfy_ui_structure: bool, is_forge_structure: bool = False, lowercase_folders: bool = False):
    """Creates the base Models directory and all predefined subdirectories, respecting ComfyUI or Forge structure if enabled."""
    if not base_path:
        print("ERROR: Base path is empty, cannot ensure directories.")
        return "Error: Base path is empty.", ["Base path is empty"]

    subdirs_to_use = get_current_subdirs(is_comfy_ui_structure, is_forge_structure)
    
    all_dirs_to_ensure = [base_path]
    for subdir_value in subdirs_to_use.values():
        resolved_full_path = resolve_target_directory(base_path, subdir_value, lowercase_folders)
        all_dirs_to_ensure.append(resolved_full_path)
    
    # Remove duplicates that might arise from resolve_target_directory if paths already exist with different casing
    all_dirs_to_ensure = sorted(list(set(all_dirs_to_ensure)))


    created_count = 0
    verified_count = 0
    errors = []

    for directory_path_str in all_dirs_to_ensure:
        try:
            # resolve_target_directory already gives the path to be created or that exists
            norm_dir = os.path.normpath(directory_path_str)
            if not os.path.exists(norm_dir):
                os.makedirs(norm_dir, exist_ok=True)
                print(f"Created directory: {norm_dir}")
                created_count += 1
            else:
                verified_count += 1
        except OSError as e:
            error_msg = f"ERROR creating directory {directory_path_str} (normalized: {norm_dir}): {str(e)}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"UNEXPECTED ERROR with directory {directory_path_str}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    status = f"Directory check complete for '{base_path}' (ComfyUI mode: {is_comfy_ui_structure}, Forge mode: {is_forge_structure}, Lowercase: {lowercase_folders}). Created: {created_count}, Verified Existing: {verified_count}."
    if errors:
        status += f" Errors: {len(errors)} (see console)."
    print(status)
    return status, errors

# --- Download Queue and Worker ---

download_queue = queue.Queue()
status_updates = queue.Queue()
stop_worker = threading.Event()
log_history = []
log_lock = threading.Lock()

def add_log(message):
    """Adds a message to the log history and prints it."""
    print(message)
    with log_lock:
        log_history.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(log_history) > 100:
            log_history.pop(0)
    if status_updates:
        try:
            log_str = "\n".join(map(str, log_history))
            status_updates.put_nowait(log_str) 
        except queue.Full:
            print("Warning: Status update queue is full, skipping update.") 
        except Exception as e:
            print(f"Error putting log update to queue: {e}")


def get_target_path(base_path: str, model_info: dict, sub_category_info: dict, is_comfy_ui_structure: bool, is_forge_structure: bool = False, lowercase_folders: bool = False) -> str:
    """Determines the full target directory path for a model, respecting ComfyUI or Forge structure."""
    subdirs_to_use = get_current_subdirs(is_comfy_ui_structure, is_forge_structure)
    target_key = model_info.get("target_dir_key") or sub_category_info.get("target_dir_key")

    if not target_key or target_key not in subdirs_to_use: # Check against current subdirs
        model_name = model_info.get('name', 'Unknown Model')
        sub_cat_name = sub_category_info.get('name', 'Unknown SubCategory') 
        if target_key:
            add_log(f"WARNING: Invalid 'target_dir_key' ('{target_key}') for {model_name} in {sub_cat_name}. Using default 'diffusion_models'.")
        else:
            add_log(f"WARNING: Missing 'target_dir_key' for {model_name} in {sub_cat_name}. Using default 'diffusion_models'.")
        target_key = "diffusion_models" 

    target_subdir_name = subdirs_to_use.get(target_key, "diffusion_models") # Get from current subdirs
    
    # Resolve the actual target directory, handling case insensitivity on Linux
    target_dir = resolve_target_directory(base_path, target_subdir_name, lowercase_folders)

    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        add_log(f"ERROR: Could not ensure target directory {target_dir} exists: {e}")
    return target_dir

def _download_model_internal(model_info, sub_category_info, base_path, use_hf_transfer, is_comfy_ui_structure, is_forge_structure=False, lowercase_folders=False):
    """Handles the download of a single model or snapshot directly to the target folder."""
    model_name = model_info.get('name', model_info.get('repo_id'))
    repo_id = model_info.get('repo_id')
    filename = model_info.get('filename_in_repo') 
    save_filename = model_info.get('save_filename') 
    is_snapshot = model_info.get('is_snapshot', False)
    allow_patterns = model_info.get('allow_patterns')
    pre_delete = model_info.get('pre_delete_target', False)
    allow_overwrite = model_info.get('allow_overwrite', False)

    if not repo_id:
        add_log(f"ERROR: Missing 'repo_id' for model {model_name}. Skipping.")
        return
    if not base_path:
        add_log(f"ERROR: Missing 'base_path' for model {model_name}. Skipping.")
        return

    target_dir = get_target_path(base_path, model_info, sub_category_info, is_comfy_ui_structure, is_forge_structure, lowercase_folders)
    if not os.path.isdir(target_dir): # Re-check after get_target_path's makedirs attempt
         add_log(f"ERROR: Target directory {target_dir} could not be confirmed for {model_name}. Skipping.")
         return

    final_target_path = os.path.join(target_dir, save_filename) if save_filename else None

    # File existence and size comparison logic
    if not is_snapshot and final_target_path and os.path.exists(final_target_path) and not allow_overwrite:
        if pre_delete:
            # For pre_delete models, always delete and re-download
            # This ensures we get the latest version of the file
            if size_data and size_data.get("models"):
                # Find the model in the structure to get category and subcategory
                for cat_name, cat_data in models_structure.items():
                    if "sub_categories" in cat_data:
                        for sub_cat_name, sub_cat_data in cat_data["sub_categories"].items():
                            for model in sub_cat_data.get("models", []):
                                if model.get("name") == model_name:
                                    model_key = f"{cat_name}::{sub_cat_name}::{model_name}"
                                    model_size_info = size_data.get("models", {}).get(model_key)
                                    if model_size_info:
                                        expected_size_gb = model_size_info.get("size_gb", 0)
                                        if expected_size_gb > 0:
                                            try:
                                                # Get actual file size
                                                actual_size_bytes = os.path.getsize(final_target_path)
                                                actual_size_gb = actual_size_bytes / (1024**3)
                                                
                                                # For pre_delete models, always delete and redownload regardless of size
                                                add_log(f"INFO: File '{final_target_path}' exists (size: {actual_size_gb:.2f} GB, expected: {expected_size_gb:.2f} GB). pre_delete is enabled, will remove and redownload.")
                                                try:
                                                    os.remove(final_target_path)
                                                    add_log(f"INFO: Removed existing file for pre_delete model: {final_target_path}")
                                                except OSError as e:
                                                    add_log(f"WARNING: Could not remove existing file {final_target_path}: {e}")
                                            except OSError as e:
                                                add_log(f"WARNING: Could not read file size for '{final_target_path}': {e}. Will proceed with redownload.")
                                                # If we can't read the file size, remove the file and proceed with download
                                                try:
                                                    os.remove(final_target_path)
                                                    add_log(f"INFO: Removed problematic file: {final_target_path}")
                                                except OSError as e2:
                                                    add_log(f"WARNING: Could not remove problematic file {final_target_path}: {e2}")
                                            except Exception as e:
                                                add_log(f"WARNING: Unexpected error during file size comparison for '{final_target_path}': {e}. Will proceed with redownload.")
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                else:
                    # No size data found for pre_delete model, proceed with redownload
                    add_log(f"INFO: No size data found for pre_delete model '{model_name}'. Will proceed with redownload.")
                    try:
                        os.remove(final_target_path)
                        add_log(f"INFO: Removed existing file for pre_delete model: {final_target_path}")
                    except OSError as e:
                        add_log(f"WARNING: Could not remove existing file {final_target_path}: {e}")
            else:
                # No size data available for pre_delete model, proceed with redownload
                add_log(f"INFO: No size data available for pre_delete model '{model_name}'. Will proceed with redownload.")
                try:
                    os.remove(final_target_path)
                    add_log(f"INFO: Removed existing file for pre_delete model: {final_target_path}")
                except OSError as e:
                    add_log(f"WARNING: Could not remove existing file {final_target_path}: {e}")
        else:
            # For non-pre_delete models, use simple existence check - skip if file exists
            add_log(f"INFO: Final target file '{final_target_path}' already exists and overwrite not allowed. Skipping download for '{model_name}'.")
            return

    if is_snapshot:
        # For snapshots, always proceed with download - snapshot_download handles skipping existing files automatically
        # This ensures new files in the repo are downloaded even if some files already exist
        if os.path.exists(target_dir):
            add_log(f"INFO: Snapshot target directory '{target_dir}' exists. Proceeding with snapshot_download (will auto-skip existing files).")
        else:
            add_log(f"INFO: Creating snapshot target directory '{target_dir}' for download.")

    add_log(f"Starting download: {model_name}...")
    try:
        start_time = time.time()
        actual_downloaded_path = None 

        if is_snapshot:
            add_log(f" -> Downloading snapshot from {repo_id} directly to {target_dir}...")
            actual_downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir, # Use resolved target_dir
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
                force_download=allow_overwrite, 
            )
            add_log(f" -> Snapshot download complete for {repo_id} into {actual_downloaded_path}.")
            final_target_path = actual_downloaded_path

        elif filename and save_filename and final_target_path:
            add_log(f" -> Downloading file '{filename}' from {repo_id} into '{target_dir}' (preserving structure from filename)...")

            force_the_download = allow_overwrite or pre_delete 

            actual_downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir, # Use resolved target_dir
                local_dir_use_symlinks=False, 
                force_download=force_the_download, 
            )
            add_log(f" -> File downloaded to actual path: {actual_downloaded_path}")

            if actual_downloaded_path != final_target_path:
                add_log(f" -> Renaming '{actual_downloaded_path}' to '{final_target_path}'...")

                os.makedirs(os.path.dirname(final_target_path), exist_ok=True)

                if os.path.exists(final_target_path):
                    if allow_overwrite or pre_delete:
                        add_log(f" -> Final target path {final_target_path} exists. Removing before rename...")
                        try:
                            if os.path.isfile(final_target_path):
                                os.remove(final_target_path)
                            else:
                                add_log(f" -> WARNING: Cannot remove final target path as it's not a file: {final_target_path}")
                                raise OSError(f"Target path for rename is not a file: {final_target_path}")
                        except OSError as e:
                            add_log(f"ERROR: Failed to remove existing file at final path '{final_target_path}' before rename: {e}. Aborting rename.")
                            raise e 
                    else:
                        add_log(f"ERROR: Final target path {final_target_path} exists and overwrite not allowed. Cannot rename. Downloaded file remains at '{actual_downloaded_path}'.")
                        return 
                try:
                    os.rename(actual_downloaded_path, final_target_path)
                    add_log(f" -> Successfully renamed to: {final_target_path}")
                except OSError as e:
                    add_log(f"ERROR: Failed to rename '{actual_downloaded_path}' to '{final_target_path}': {e}")
                    add_log(f" -> The originally downloaded file likely remains at: {actual_downloaded_path}")
                    raise e 
            else:
                add_log(f" -> Actual download path matches desired final path. No rename needed.")

        else:
             if is_snapshot: 
                 add_log(f"ERROR: Internal logic error for snapshot {model_name}. Skipping.")
             elif not filename:
                  add_log(f"ERROR: Invalid configuration for model {model_name}. Missing 'filename_in_repo'. Skipping.")
             elif not save_filename:
                  add_log(f"ERROR: Invalid configuration for model {model_name}. Missing 'save_filename'. Skipping.")
             else:
                  add_log(f"ERROR: Invalid configuration for model {model_name}. Path issue? Skipping.")
             return 

        end_time = time.time()
        success_path = final_target_path if not is_snapshot else actual_downloaded_path 
        add_log(f"SUCCESS: Downloaded and processed {model_name} in {end_time - start_time:.2f} seconds. Final location: {success_path}")

    except (HfHubHTTPError, HFValidationError) as e:
        add_log(f"ERROR downloading {model_name} (HF Hub): {type(e).__name__} - {str(e)}")
    except FileNotFoundError as e:
         add_log(f"ERROR during file operation for {model_name} (File System): {type(e).__name__} - {str(e)}")
    except OSError as e:
         add_log(f"ERROR during file operation (rename/delete) for {model_name} (OS Error/Permissions): {type(e).__name__} - {str(e)}")
    except Exception as e:
        add_log(f"UNEXPECTED ERROR during download/process for {model_name}: {type(e).__name__} - {str(e)}")
        if 'actual_downloaded_path' in locals() and actual_downloaded_path:
             add_log(f" -> State before error: actual_downloaded_path='{actual_downloaded_path}'")
        if 'final_target_path' in locals() and final_target_path:
             add_log(f" -> State before error: final_target_path='{final_target_path}'")

def download_worker():
    """Worker thread function to process the download queue."""
    print("Download worker thread started.")
    while not stop_worker.is_set():
        try:
            task = download_queue.get(timeout=1)
        except queue.Empty:
            continue

        model_info, sub_category_info, base_path, use_hf_transfer, is_comfy_ui_structure, is_forge_structure, lowercase_folders = task
        original_hf_transfer_env = None
        try:
            original_hf_transfer_env = os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')
            transfer_env_value = '1' if use_hf_transfer and HF_TRANSFER_AVAILABLE else '0'
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = transfer_env_value
            
            _download_model_internal(model_info, sub_category_info, base_path, use_hf_transfer, is_comfy_ui_structure, is_forge_structure, lowercase_folders)

        except Exception as e:
            model_name_for_log = model_info.get('name', 'unknown task')
            add_log(f"CRITICAL WORKER ERROR processing '{model_name_for_log}': {type(e).__name__} - {e}")
        finally:
            if original_hf_transfer_env is None:
                if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
                    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']
            else:
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = original_hf_transfer_env
            download_queue.task_done()
    print("Download worker thread stopped.")


# --- Filtering Logic ---
# No changes needed in filter_models for this request.
def filter_models(structure, search_term):
    search_term = search_term.lower().strip()
    visibility = {} 
    if not search_term:
        for cat_name, cat_data in structure.items():
            cat_key = f"cat_{cat_name}"
            visibility[cat_key] = True
            if "sub_categories" in cat_data:
                for sub_cat_name in cat_data["sub_categories"]:
                    sub_cat_key = f"subcat_{cat_name}_{sub_cat_name}"
                    visibility[sub_cat_key] = True
            elif "bundles" in cat_data:
                for i, bundle_data in enumerate(cat_data["bundles"]):
                     bundle_key = f"bundle_{cat_name}_{i}"
                     visibility[bundle_key] = True
                     bundle_button_key = f"bundlebutton_{cat_name}_{i}"
                     visibility[bundle_button_key] = True
        return visibility
    for cat_name, cat_data in structure.items():
        cat_key = f"cat_{cat_name}"
        visibility[cat_key] = False 
        if "sub_categories" in cat_data:
            for sub_cat_name in cat_data["sub_categories"]:
                sub_cat_key = f"subcat_{cat_name}_{sub_cat_name}"
                visibility[sub_cat_key] = False
        elif "bundles" in cat_data:
             for i, bundle_data in enumerate(cat_data["bundles"]):
                 bundle_key = f"bundle_{cat_name}_{i}"
                 visibility[bundle_key] = False
                 bundle_button_key = f"bundlebutton_{cat_name}_{i}"
                 visibility[bundle_button_key] = False
    for cat_name, cat_data in structure.items():
        cat_key = f"cat_{cat_name}"
        cat_match = search_term in cat_name.lower()
        cat_becomes_visible = cat_match 
        if "sub_categories" in cat_data:
            for sub_cat_name, sub_cat_data in cat_data["sub_categories"].items():
                sub_cat_key = f"subcat_{cat_name}_{sub_cat_name}"
                sub_cat_match = search_term in sub_cat_name.lower()
                sub_cat_becomes_visible = sub_cat_match 
                model_match_found = False
                for model_info in sub_cat_data.get("models", []):
                    model_name = model_info.get("name", "").lower()
                    if search_term in model_name:
                        model_match_found = True
                        break 
                if model_match_found or sub_cat_match:
                    visibility[sub_cat_key] = True 
                    cat_becomes_visible = True 
            if cat_becomes_visible:
                 visibility[cat_key] = True 
        elif "bundles" in cat_data:
             bundle_match_found_in_cat = False
             for i, bundle_data in enumerate(cat_data["bundles"]):
                 bundle_key = f"bundle_{cat_name}_{i}"
                 bundle_button_key = f"bundlebutton_{cat_name}_{i}"
                 bundle_name = bundle_data.get("name", "").lower()
                 bundle_info = bundle_data.get("info", "").lower() 
                 if search_term in bundle_name or search_term in bundle_info:
                     visibility[bundle_key] = True
                     visibility[bundle_button_key] = True
                     bundle_match_found_in_cat = True
             if bundle_match_found_in_cat or cat_match: 
                 visibility[cat_key] = True
    return visibility

# --- Bundle Helper ---
# No changes needed in find_model_by_key for this request.
def find_model_by_key(category_name, sub_category_name, model_name):
    try:
        category_data = models_structure[category_name]
        sub_category_data = category_data["sub_categories"][sub_category_name]
        for model_info in sub_category_data["models"]:
            if model_info["name"] == model_name:
                return model_info, sub_category_data
        add_log(f"ERROR: Model '{model_name}' not found in '{category_name}' -> '{sub_category_name}'.")
        return None, None
    except KeyError:
        add_log(f"ERROR: Category '{category_name}' or Sub-category '{sub_category_name}' not found while searching for model '{model_name}'.")
        return None, None
    except Exception as e:
        add_log(f"ERROR: Unexpected error finding model '{model_name}': {e}")
        return None, None


# --- Gradio UI Builder ---

def create_ui(default_base_path):
    """Creates the Gradio interface."""
    # Load model size data
    has_size_data = load_model_sizes()
    
    tracked_components = {}
    tracked_accordions = {}  # New: Track all accordion components for expand/collapse functionality

    with gr.Blocks(theme=gr.themes.Soft(), title=APP_TITLE, css="""
    /* TESTING: Make all buttons left-aligned to see if CSS works at all */
    button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Target Gradio buttons with left-aligned-button class */
    .left-aligned-button,
    .left-aligned-button > *,
    [class*="left-aligned-button"],
    [class*="left-aligned-button"] button,
    [class*="left-aligned-button"] .gr-button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Target specific button by ID */
    #left-aligned-bundle-button,
    #left-aligned-bundle-button button,
    #left-aligned-bundle-button * {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* More specific targeting for button content */
    .left-aligned-button button,
    .left-aligned-button .gr-button,
    #left-aligned-bundle-button button {
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 12px !important;
    }
    
    /* Target button text specifically */
    .left-aligned-button button *,
    .left-aligned-button .gr-button *,
    #left-aligned-bundle-button * {
        text-align: left !important;
        justify-self: flex-start !important;
    }
    
    /* Very aggressive approach - target all buttons if needed */
    button[class*="left-aligned"],
    .gradio-container button.left-aligned-button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Hint text styling */
    .hint-text {
        font-size: 0.9em !important;
        color: #666 !important;
        margin-top: 5px !important;
        margin-bottom: 10px !important;
    }
    
    /* Expand/Collapse buttons styling */
    .expand-collapse-buttons {
        margin-bottom: 15px !important;
    }
    """) as app:
        gr.Markdown(f"## {APP_TITLE} V80 > Source : https://www.patreon.com/posts/114517862")
        gr.Markdown(f"### ComfyUI Installer for SwarmUI's Back-End > https://www.patreon.com/posts/105023709")
        gr.Markdown(f"### 5 May 2025 Main How To Install & Use Tutorial : https://youtu.be/fTzlQ0tjxj0")   
        gr.Markdown(f"### 17 June 2025 WAN 2.1 FusionX is the New Best of Local Video Generation with Only 8 Steps + FLUX Upscaling Guide Tutorial : https://youtu.be/Xbn93GRQKsQ")    
        gr.Markdown(f"### 2 August 2025 Wan 2.2 & FLUX Krea Full Tutorial - Automated Install - Ready Perfect Presets - SwarmUI with ComfyUI Tutorial : https://youtu.be/8MvvuX4YPeo")         
        gr.Markdown("### Select models or bundles to download. Downloads will be added to a queue. Use the search bar to filter.")
        
        log_output = gr.Textbox(label="Download Status / Log - Watch CMD / Terminal To See Download Status & Speed", lines=10, max_lines=20, interactive=False, value="Welcome! Logs will appear here.")
        queue_status_label = gr.Markdown(f"Queue Size: {download_queue.qsize()}")

        with gr.Row():
             search_box = gr.Textbox(placeholder="Search models or bundles...", label="Search", scale=2, interactive=True)
             use_hf_transfer_checkbox = gr.Checkbox(label="Enable hf_transfer (Faster Downloads)", value=HF_TRANSFER_AVAILABLE, scale=1)
        
        with gr.Row():
             base_path_input = gr.Textbox(label="Base Download Path (SwarmUI/Models)", value=default_base_path, scale=3)
             comfy_ui_structure_checkbox = gr.Checkbox(label="ComfyUI Folder Structure (e.g. 'loras' folder)", value=get_default_comfy_ui_structure(), scale=1)
             forge_structure_checkbox = gr.Checkbox(label="Forge WebUI / Automatic1111 Folder Structure", value=get_default_forge_structure(), scale=1)
             lowercase_folders_checkbox = gr.Checkbox(label="Lowercase Folder Names", value=get_default_lowercase_folders(), scale=1)
             remember_path_button = gr.Button("💾 Remember Settings", scale=1, size="sm")
        
        remember_path_status = gr.Markdown("", visible=False)
        
        # Consolidated info and buttons in one row
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("💡 **Tip:** Use 'Remember Settings' to save your preferred download location, ComfyUI/Forge structure, and lowercase folder preference. • **Note:** Only one structure (ComfyUI or Forge) can be active at a time. • **Lowercase Folders:** When enabled, folder names convert to lowercase (e.g., 'Stable-Diffusion' → 'stable-diffusion').", elem_classes="hint-text")
            expand_all_button = gr.Button("📂 Expand All", size="sm", scale=1)
            collapse_all_button = gr.Button("📁 Collapse All", size="sm", scale=1)
        
        # Search results container for direct download buttons
        with gr.Column(visible=False) as search_results_container:
            gr.Markdown("### Search Results")
            search_results_message = gr.Markdown("")
            
            # Pre-create search result rows (we'll show/hide them dynamically)
            MAX_SEARCH_RESULTS = 50
            search_result_rows = []
            search_result_buttons = []
            
            for i in range(MAX_SEARCH_RESULTS):
                with gr.Row(visible=False) as row:
                    download_btn = gr.Button("", elem_classes="left-aligned-button")
                    
                search_result_rows.append({
                    "row": row,
                    "button": download_btn
                })
        
        # Functions to handle expand/collapse all
        def expand_all_accordions():
            """Expand all accordions"""
            updates = []
            for accordion in tracked_accordions.values():
                updates.append(gr.update(open=True))
            add_log("Expanded all accordions")
            return updates
        
        def collapse_all_accordions():
            """Collapse all accordions"""
            updates = []
            for accordion in tracked_accordions.values():
                updates.append(gr.update(open=False))
            add_log("Collapsed all accordions")
            return updates
        
        # We'll connect these buttons after creating all accordions

        # Initial directory check with default ComfyUI structure (False)
        # initial_dir_status, _ = ensure_directories_exist(default_base_path, False) # Pass comfy_ui_structure_checkbox.value (default False)
        # add_log(f"Initial directory check: {initial_dir_status}")

        def update_hf_transfer_setting(value):
            add_log(f"User {'enabled' if value else 'disabled'} hf_transfer checkbox.")

        use_hf_transfer_checkbox.change(fn=update_hf_transfer_setting, inputs=use_hf_transfer_checkbox, outputs=None)

        def handle_remember_path_click(current_path, comfy_ui_checked, forge_checked, lowercase_folders_checked):
            """Handles the remember path button click."""
            if not current_path or not current_path.strip():
                return gr.update(value="⚠️ Please enter a path first", visible=True)
            
            result = save_last_settings(current_path.strip(), comfy_ui_checked, forge_checked, lowercase_folders_checked)
            return gr.update(value=result, visible=True)

        remember_path_button.click(
            fn=handle_remember_path_click,
            inputs=[base_path_input, comfy_ui_structure_checkbox, forge_structure_checkbox, lowercase_folders_checkbox],
            outputs=[remember_path_status]
        )

        def handle_comfy_checkbox_change(is_comfy_checked):
            """Handle ComfyUI checkbox change and uncheck Forge if ComfyUI is checked."""
            if is_comfy_checked:
                add_log("ComfyUI folder structure enabled, disabling Forge structure.")
                return gr.update(value=False)  # Uncheck Forge
            return gr.update()  # No change to Forge
        
        def handle_forge_checkbox_change(is_forge_checked):
            """Handle Forge checkbox change and uncheck ComfyUI if Forge is checked."""
            if is_forge_checked:
                add_log("Forge folder structure enabled, disabling ComfyUI structure.")
                return gr.update(value=False)  # Uncheck ComfyUI
            return gr.update()  # No change to ComfyUI

        def handle_dir_structure_change(current_base_path, is_comfy_checked, is_forge_checked, lowercase_folders_checked):
            # status_msg, _ = ensure_directories_exist(current_base_path, is_comfy_checked, is_forge_checked, lowercase_folders_checked)
            add_log(f"Base path or structure setting changed. Base: '{current_base_path}', ComfyUI Mode: {is_comfy_checked}, Forge Mode: {is_forge_checked}, Lowercase: {lowercase_folders_checked}. Directories will be ensured upon download initiation.")
            # No direct output to UI component from here, log is sufficient.
            # Or return status_msg to a dedicated status gr.Markdown if needed.

        # Handle mutual exclusivity between ComfyUI and Forge checkboxes
        comfy_ui_structure_checkbox.change(
            fn=handle_comfy_checkbox_change,
            inputs=[comfy_ui_structure_checkbox],
            outputs=[forge_structure_checkbox]
        )
        forge_structure_checkbox.change(
            fn=handle_forge_checkbox_change,
            inputs=[forge_structure_checkbox],
            outputs=[comfy_ui_structure_checkbox]
        )
        
        # Handle directory structure changes
        comfy_ui_structure_checkbox.change(
            fn=handle_dir_structure_change,
            inputs=[base_path_input, comfy_ui_structure_checkbox, forge_structure_checkbox, lowercase_folders_checkbox],
            outputs=None # Log output is handled by add_log
        )
        forge_structure_checkbox.change(
            fn=handle_dir_structure_change,
            inputs=[base_path_input, comfy_ui_structure_checkbox, forge_structure_checkbox, lowercase_folders_checkbox],
            outputs=None # Log output is handled by add_log
        )
        base_path_input.change( # Assuming base_path_input doesn't have other .change events that would conflict. If so, combine logic.
            fn=handle_dir_structure_change,
            inputs=[base_path_input, comfy_ui_structure_checkbox, forge_structure_checkbox, lowercase_folders_checkbox],
            outputs=None # Log output is handled by add_log
        )
        lowercase_folders_checkbox.change(
            fn=handle_dir_structure_change,
            inputs=[base_path_input, comfy_ui_structure_checkbox, forge_structure_checkbox, lowercase_folders_checkbox],
            outputs=None # Log output is handled by add_log
        )

        def enqueue_download(model_info, sub_category_info, current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders):
            if not current_base_path:
                 add_log("ERROR: Cannot queue download, base path input is empty.")
                 return f"Queue Size: {download_queue.qsize()}"
            if not isinstance(sub_category_info, dict):
                add_log(f"ERROR: Invalid sub_category_info type ({type(sub_category_info)}) for model {model_info.get('name')}. Skipping queue.")
                return f"Queue Size: {download_queue.qsize()}"

            dir_status_msg, dir_errors = ensure_directories_exist(current_base_path, is_comfy_checked, is_forge_checked, lowercase_folders)
            add_log(f"Directory check for download: {dir_status_msg}")
            if dir_errors:
                add_log(f"ERROR: Directory setup failed for '{current_base_path}'. Download of '{model_info.get('name', model_info.get('repo_id'))}' aborted.")
                for err in dir_errors:
                    add_log(f"  - {err}")
                return f"Queue Size: {download_queue.qsize()}"

            download_queue.put((model_info, sub_category_info, current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders))
            add_log(f"Queued: {model_info.get('name', model_info.get('repo_id'))}")
            return f"Queue Size: {download_queue.qsize()}"

        def enqueue_bulk_download(models_list, sub_category_info, current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders):
            if not current_base_path:
                 add_log("ERROR: Cannot queue bulk download, base path input is empty.")
                 return f"Queue Size: {download_queue.qsize()}"
            if not isinstance(sub_category_info, dict):
                add_log(f"ERROR: Invalid sub_category_info type ({type(sub_category_info)}) for bulk download. Skipping queue.")
                return f"Queue Size: {download_queue.qsize()}"

            dir_status_msg, dir_errors = ensure_directories_exist(current_base_path, is_comfy_checked, is_forge_checked, lowercase_folders)
            add_log(f"Directory check for bulk download: {dir_status_msg}")
            if dir_errors:
                sub_cat_name_log = sub_category_info.get("name", "Group")
                add_log(f"ERROR: Directory setup failed for '{current_base_path}'. Bulk download from '{sub_cat_name_log}' aborted.")
                for err in dir_errors:
                    add_log(f"  - {err}")
                return f"Queue Size: {download_queue.qsize()}"

            count = 0
            sub_cat_name = sub_category_info.get("name", "Group") 
            for model_info in models_list:
                 download_queue.put((model_info, sub_category_info, current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders))
                 count += 1
            add_log(f"Queued {count} models from '{sub_cat_name}'.")
            return f"Queue Size: {download_queue.qsize()}"

        def enqueue_bundle_download(bundle_definition, current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders):
            if not current_base_path:
                add_log("ERROR: Cannot queue bundle download, base path input is empty.")
                return f"Queue Size: {download_queue.qsize()}"

            bundle_name = bundle_definition.get("name", "Unnamed Bundle")
            dir_status_msg, dir_errors = ensure_directories_exist(current_base_path, is_comfy_checked, is_forge_checked, lowercase_folders)
            add_log(f"Directory check for bundle '{bundle_name}': {dir_status_msg}")
            if dir_errors:
                add_log(f"ERROR: Directory setup failed for '{current_base_path}'. Bundle download '{bundle_name}' aborted.")
                for err in dir_errors:
                    add_log(f"  - {err}")
                return f"Queue Size: {download_queue.qsize()}"

            model_keys = bundle_definition.get("models_to_download", [])
            queued_count = 0
            errors = 0

            add_log(f"Queueing bundle: '{bundle_name}'...")
            for cat_name, sub_cat_name, model_name in model_keys:
                model_info, sub_cat_info = find_model_by_key(cat_name, sub_cat_name, model_name)
                if model_info and sub_cat_info:
                    # Use the standard enqueue function, passing comfy_checked state
                    enqueue_download(model_info, sub_cat_info, current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders)
                    queued_count += 1
                else:
                    errors += 1
                    add_log(f"  -> ERROR: Could not find model '{model_name}' for bundle. Skipping.")

            add_log(f"Bundle '{bundle_name}' processed. Queued: {queued_count}, Errors: {errors}.")
            return f"Queue Size: {download_queue.qsize()}"

        for cat_name, cat_data in models_structure.items():
            cat_key = f"cat_{cat_name}"
            with gr.Accordion(cat_name, open=False, visible=True) as cat_accordion: 
                tracked_components[cat_key] = cat_accordion
                tracked_accordions[f"cat_{cat_name}"] = cat_accordion  # Track for expand/collapse
                
                if "bundles" in cat_data:
                    gr.Markdown(cat_data.get("info", ""))
                    for i, bundle_info in enumerate(cat_data.get("bundles", [])):
                        bundle_key = f"bundle_{cat_name}_{i}"
                        bundle_button_key = f"bundlebutton_{cat_name}_{i}"
                        bundle_display_name = bundle_info.get("name", f"Bundle {i+1}")
                        bundle_size_display = get_bundle_size_display(cat_name, i)
                        with gr.Column(variant="panel", visible=True) as bundle_container:
                            tracked_components[bundle_key] = bundle_container 
                            gr.Markdown(f"**{bundle_display_name}{bundle_size_display}**")
                            
                            # Get bundle info with sizes integrated into the Includes section
                            bundle_base_info = bundle_info.get("info", "*No description provided.*")
                            full_bundle_info = get_bundle_with_sizes_info(cat_name, i, bundle_base_info)
                            gr.Markdown(full_bundle_info)
                            
                            download_bundle_button = gr.Button(f"Download {bundle_display_name}{bundle_size_display}", elem_id="left-aligned-bundle-button")
                            tracked_components[bundle_button_key] = download_bundle_button 
                            download_bundle_button.click(
                                fn=enqueue_bundle_download,
                                inputs=[
                                    gr.State(bundle_info), 
                                    base_path_input,
                                    use_hf_transfer_checkbox,
                                    comfy_ui_structure_checkbox, # Pass new checkbox state
                                    forge_structure_checkbox,
                                    lowercase_folders_checkbox
                                ],
                                outputs=[queue_status_label]
                             )
                elif "sub_categories" in cat_data:
                    gr.Markdown(cat_data.get("info", ""))
                    for sub_cat_name, sub_cat_data in cat_data.get("sub_categories", {}).items():
                        sub_cat_key = f"subcat_{cat_name}_{sub_cat_name}"
                        with gr.Column(variant="panel", elem_classes="sub-category-panel", visible=True) as subcat_container: 
                            tracked_components[sub_cat_key] = subcat_container 
                            with gr.Accordion(sub_cat_name, open=False) as subcat_accordion:
                                tracked_accordions[f"subcat_{cat_name}_{sub_cat_name}"] = subcat_accordion  # Track for expand/collapse
                                
                                gr.Markdown(sub_cat_data.get("info", ""))
                                models_in_subcat = sub_cat_data.get("models", [])
                                if not models_in_subcat:
                                    gr.Markdown("*No models listed in this sub-category yet.*")
                                    continue
                                
                                for model_info in models_in_subcat:
                                    model_display_name = model_info.get("name", "Unknown Model")
                                    model_size_display = get_model_size_display(cat_name, sub_cat_name, model_display_name)
                                    button_text = f"- {model_display_name}{model_size_display}"
                                    
                                    download_button = gr.Button(button_text, elem_classes="left-aligned-button")
                                    
                                    # Prepare state for the click handler
                                    # Create a fresh copy of sub_cat_data for each button's state
                                    # to ensure 'name' is correctly set if missing from original sub_cat_data
                                    current_sub_cat_state_data = sub_cat_data.copy()
                                    if 'name' not in current_sub_cat_state_data:
                                        current_sub_cat_state_data['name'] = sub_cat_name
                                    
                                    download_button.click(
                                        fn=enqueue_download,
                                        inputs=[
                                            gr.State(model_info),
                                            gr.State(current_sub_cat_state_data), 
                                            base_path_input,
                                            use_hf_transfer_checkbox,
                                            comfy_ui_structure_checkbox,
                                            forge_structure_checkbox,
                                            lowercase_folders_checkbox
                                        ],
                                        outputs=[queue_status_label]
                                    )
                                    
                                if models_in_subcat: # "Download All" button section
                                     with gr.Row():
                                         gr.Markdown("---")
                                     with gr.Row():
                                         subcat_size_display = get_subcategory_total_size_display(cat_name, sub_cat_name, models_in_subcat)
                                         download_all_button = gr.Button(f"Download All {sub_cat_name}{subcat_size_display}", elem_classes="left-aligned-button")
                                         # Prepare state for the "Download All" button's click handler
                                         # Similar to individual buttons, ensure 'name' is in the state
                                         all_sub_cat_state_data = sub_cat_data.copy()
                                         if 'name' not in all_sub_cat_state_data:
                                             all_sub_cat_state_data['name'] = sub_cat_name
                                         download_all_button.click(
                                             fn=enqueue_bulk_download,
                                             inputs=[
                                                 gr.State(models_in_subcat),
                                                 gr.State(all_sub_cat_state_data), 
                                                 base_path_input,
                                                 use_hf_transfer_checkbox,
                                                 comfy_ui_structure_checkbox,
                                                 forge_structure_checkbox,
                                                 lowercase_folders_checkbox
                                             ],
                                             outputs=[queue_status_label]
                                         )
                else:
                     gr.Markdown(cat_data.get("info", "*No sub-categories or bundles defined.*"))

        # Connect expand/collapse buttons after all accordions are created
        expand_all_button.click(
            fn=expand_all_accordions,
            inputs=[],
            outputs=list(tracked_accordions.values())
        )
        
        collapse_all_button.click(
            fn=collapse_all_accordions,
            inputs=[],
            outputs=list(tracked_accordions.values())
        )

        # Store search results data for button handlers
        search_results_data = []
        
        def update_search_results(search_term: str):
            """Update search results display"""
            if not search_term or not search_term.strip():
                # Hide search results and show normal view
                updates = [gr.update(visible=False), gr.update()]  # search container, message
                
                # Show all normal components
                for key, component in tracked_components.items():
                    updates.append(gr.update(visible=True))
                
                # Hide all search result rows
                for row_data in search_result_rows:
                    updates.append(gr.update(visible=False))  # row
                    updates.append(gr.update())  # button
                    
                return updates
            
            search_term_lower = search_term.lower().strip()
            matching_items = []
            
            # Search for matching models
            for cat_name, cat_data in models_structure.items():
                if "sub_categories" in cat_data:
                    for sub_cat_name, sub_cat_data in cat_data["sub_categories"].items():
                        for model_info in sub_cat_data.get("models", []):
                            model_name = model_info.get("name", "")
                            if search_term_lower in model_name.lower():
                                matching_items.append({
                                    "type": "model",
                                    "model": model_info,
                                    "category": cat_name,
                                    "subcategory": sub_cat_name,
                                    "sub_cat_data": sub_cat_data
                                })
                
                # Search for matching bundles
                if "bundles" in cat_data:
                    for i, bundle_info in enumerate(cat_data.get("bundles", [])):
                        bundle_name = bundle_info.get("name", "")
                        bundle_info_text = bundle_info.get("info", "")
                        if search_term_lower in bundle_name.lower() or search_term_lower in bundle_info_text.lower():
                            matching_items.append({
                                "type": "bundle",
                                "bundle": bundle_info,
                                "category": cat_name,
                                "index": i
                            })
            
            # Clear search results data
            search_results_data.clear()
            search_results_data.extend(matching_items)
            
            # Prepare updates
            updates = []
            
            # Show search container
            updates.append(gr.update(visible=True))
            
            # Update message
            if not matching_items:
                updates.append(gr.update(value=f"No results found for '{search_term}'"))
            else:
                model_count = sum(1 for item in matching_items if item["type"] == "model")
                bundle_count = sum(1 for item in matching_items if item["type"] == "bundle")
                msg = f"Found {len(matching_items)} results: "
                if model_count > 0:
                    msg += f"{model_count} model{'s' if model_count > 1 else ''}"
                if bundle_count > 0:
                    if model_count > 0:
                        msg += f", "
                    msg += f"{bundle_count} bundle{'s' if bundle_count > 1 else ''}"
                updates.append(gr.update(value=msg))
            
            # Hide all normal components
            for key, component in tracked_components.items():
                updates.append(gr.update(visible=False))
            
            # Update search result rows
            for i, row_data in enumerate(search_result_rows):
                if i < len(matching_items):
                    item = matching_items[i]
                    
                    if item["type"] == "model":
                        model_info = item["model"]
                        cat_name = item["category"]
                        sub_cat_name = item["subcategory"]
                        
                        model_display_name = model_info.get("name", "Unknown Model")
                        model_size_display = get_model_size_display(cat_name, sub_cat_name, model_display_name)
                        
                        button_text = f"- {model_display_name}{model_size_display}"
                    else:  # bundle
                        bundle_info = item["bundle"]
                        cat_name = item["category"]
                        bundle_index = item["index"]
                        
                        bundle_name = bundle_info.get("name", "Unknown Bundle")
                        bundle_size_display = get_bundle_size_display(cat_name, bundle_index)
                        
                        button_text = f"Download {bundle_name}{bundle_size_display}"
                    
                    updates.append(gr.update(visible=True))  # row
                    updates.append(gr.update(value=button_text))  # button
                else:
                    updates.append(gr.update(visible=False))  # row
                    updates.append(gr.update())  # button
            
            return updates
        
        # Create button click handlers for search results
        def create_download_handler(index):
            def handler(current_base_path, hf_transfer_enabled, is_comfy_checked, is_forge_checked, lowercase_folders):
                if index >= len(search_results_data):
                    return gr.update()
                    
                item = search_results_data[index]
                
                if item["type"] == "model":
                    model_info = item["model"]
                    sub_cat_data = item["sub_cat_data"]
                    sub_cat_name = item["subcategory"]
                    
                    # Prepare state data
                    current_sub_cat_state_data = sub_cat_data.copy()
                    if 'name' not in current_sub_cat_state_data:
                        current_sub_cat_state_data['name'] = sub_cat_name
                    
                    return enqueue_download(
                        model_info,
                        current_sub_cat_state_data,
                        current_base_path,
                        hf_transfer_enabled,
                        is_comfy_checked,
                        is_forge_checked,
                        lowercase_folders
                    )
                else:  # bundle
                    bundle_info = item["bundle"]
                    
                    return enqueue_bundle_download(
                        bundle_info,
                        current_base_path,
                        hf_transfer_enabled,
                        is_comfy_checked,
                        is_forge_checked,
                        lowercase_folders
                    )
            return handler
        
        # Connect search result buttons
        for i, row_data in enumerate(search_result_rows):
            row_data["button"].click(
                fn=create_download_handler(i),
                inputs=[
                    base_path_input,
                    use_hf_transfer_checkbox,
                    comfy_ui_structure_checkbox,
                    forge_structure_checkbox,
                    lowercase_folders_checkbox
                ],
                outputs=[queue_status_label]
            )
        
        # Create outputs list for search change handler
        search_outputs = [search_results_container, search_results_message]
        search_outputs.extend(list(tracked_components.values()))
        for row_data in search_result_rows:
            search_outputs.extend([row_data["row"], row_data["button"]])
        
        search_box.change(
            fn=update_search_results,
            inputs=[search_box],
            outputs=search_outputs
        )

        try:
            timer = gr.Timer(1, active=True) 
            def update_log_display():
                log_update = gr.update() 
                queue_update = gr.update() 
                new_log_available = False
                try:
                    latest_log = status_updates.get_nowait()
                    log_update = latest_log 
                    new_log_available = True
                except queue.Empty:
                    pass 
                q_size = download_queue.qsize()
                queue_update = f"Queue Size: {q_size}"
                return log_update, queue_update
            timer.tick(update_log_display, None, [log_output, queue_status_label])
            add_log("Using gr.Timer for UI updates.")
        except AttributeError:
            add_log("gr.Timer not found, falling back to deprecated app.load(every=1) for UI updates.")
            def update_log_display_legacy():
                 log_update = gr.update()
                 queue_update = gr.update()
                 try:
                     latest_log = status_updates.get_nowait()
                     log_update = latest_log
                 except queue.Empty:
                     pass
                 q_size = download_queue.qsize()
                 queue_update = f"Queue Size: {q_size}"
                 return {log_output: log_update, queue_status_label: queue_update}
            app.load(update_log_display_legacy, None, [log_output, queue_status_label], every=1)
    return app

# --- Main Execution ---

def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        try:
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1: drives.append(f"{letter}:\\")
                bitmask >>= 1
            available_paths = drives
        except Exception as e:
            print(f"Warning: Could not get Windows drives via ctypes: {e}")
            available_paths = ["C:\\"] # Fallback
    elif platform.system() == "Darwin":
         available_paths = ["/", "/Volumes"]
    else: # Linux/Other
        available_paths = ["/", "/mnt", "/media", "/run/media"] 

    existing_paths = [p for p in available_paths if os.path.isdir(p)]
    try:
        home_dir = os.path.expanduser("~")
        if os.path.isdir(home_dir) and home_dir not in existing_paths:
            is_sub = False
            for p in existing_paths:
                try:
                    norm_p = os.path.normpath(p)
                    norm_home = os.path.normpath(home_dir)
                    if os.path.commonpath([norm_p, norm_home]) == norm_p:
                        is_sub = True
                        break
                except ValueError: pass 
                except Exception: pass 
            if not is_sub:
                existing_paths.append(home_dir)
    except Exception as e:
        print(f"Warning: Could not reliably determine home directory: {e}")
    try:
        cwd = os.getcwd()
        is_subpath = False
        for p in existing_paths:
            try:
                norm_p = os.path.normpath(p)
                norm_cwd = os.path.normpath(cwd)
                if os.path.commonpath([norm_p, norm_cwd]) == norm_p:
                     is_subpath = True
                     break
            except ValueError: pass 
            except Exception as e:
                 print(f"Warning: Error checking common path for {p} and {cwd}: {e}")
        if not is_subpath and os.path.isdir(cwd) and cwd not in existing_paths:
             existing_paths.append(cwd)
    except Exception as e:
        print(f"Warning: Could not reliably determine current working directory: {e}")
    print(f"Detected potential root paths: {existing_paths}")
    return existing_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmUI Model Downloader - Direct Download Version with Search and Bundles")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link")
    parser.add_argument("--model-path", type=str, default=None, help="Override default SwarmUI Models path")
    args = parser.parse_args()

    if args.model_path:
        current_base_path = os.path.abspath(args.model_path)
        print(f"Using base path from command line: {current_base_path}")
    else:
        current_base_path = os.path.abspath(DEFAULT_BASE_PATH) 
        saved_path, saved_comfy_ui_structure, saved_forge_structure, saved_lowercase_folders = load_last_settings()
        if saved_path:
            print(f"Using saved settings from previous session: Path='{current_base_path}', ComfyUI={saved_comfy_ui_structure}, Forge={saved_forge_structure}, Lowercase={saved_lowercase_folders}")
        else:
            print(f"Using default base path: {current_base_path}")

    # Ensure Base Dirs Exist Early (default ComfyUI mode to False for this initial call)
    # ensure_directories_exist(current_base_path, False) 

    worker_thread = threading.Thread(target=download_worker, daemon=True)
    worker_thread.start()

    gradio_app = create_ui(current_base_path)
    allowed_paths_list = get_available_drives()
    try:
        base_dir_norm = os.path.normpath(current_base_path)
        parent_dir_norm = os.path.normpath(os.path.dirname(base_dir_norm))
        def is_subpath_of_allowed(path_to_check, allowed_list):
            norm_check = os.path.normpath(path_to_check)
            for allowed in allowed_list:
                norm_allowed = os.path.normpath(allowed)
                try:
                    if os.path.commonpath([norm_allowed, norm_check]) == norm_allowed:
                        return True
                except ValueError: 
                    pass
                except Exception as e:
                    print(f"Warning: Error checking common path for {norm_allowed} and {norm_check}: {e}")
            return False
        if os.path.isdir(base_dir_norm) and not is_subpath_of_allowed(base_dir_norm, allowed_paths_list):
            allowed_paths_list.append(base_dir_norm)
        if os.path.isdir(parent_dir_norm) and parent_dir_norm != base_dir_norm and not is_subpath_of_allowed(parent_dir_norm, allowed_paths_list):
            allowed_paths_list.append(parent_dir_norm)
    except Exception as e:
        print(f"Warning: Error processing base/parent paths for Gradio allowed_paths: {e}")
        if os.path.isdir(current_base_path) and current_base_path not in allowed_paths_list:
             allowed_paths_list.append(current_base_path)
    print(f"Final allowed Gradio paths for launch: {allowed_paths_list}")

    try:
        gradio_app.launch(
            inbrowser=True,
            share=args.share,
            allowed_paths=allowed_paths_list
        )
    except KeyboardInterrupt:
        print("\nCtrl+C received. Shutting down...")
    except Exception as e:
         print(f"ERROR launching Gradio: {e}")
         print("Please ensure Gradio is installed correctly (`pip install gradio`) and that the specified port is available.")
    finally:
        stop_worker.set()
        print("Waiting for download worker to finish current task (up to 5s)...")
        worker_thread.join(timeout=5.0) 
        if worker_thread.is_alive():
            print("Worker thread did not finish cleanly after 5 seconds.")
        else:
            print("Download worker stopped.")
        if status_updates is not None:
             status_updates.put(None) 
             status_updates = None
    print("Gradio app closed.")