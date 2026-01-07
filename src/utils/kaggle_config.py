"""
Kaggle Configuration
====================

Configuration for running TriMedAgent on Kaggle with 2xT4 GPUs.
"""

import os
import torch
from pathlib import Path

# =============================================================================
# Environment Detection
# =============================================================================

def is_kaggle():
    """Check if running on Kaggle."""
    return os.path.exists("/kaggle/working")

def is_colab():
    """Check if running on Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_environment():
    """Get current environment name."""
    if is_kaggle():
        return "kaggle"
    elif is_colab():
        return "colab"
    else:
        return "local"


# =============================================================================
# GPU Configuration
# =============================================================================

def get_gpu_count():
    """Get number of available GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def get_device_map():
    """
    Get optimal device mapping for 2xT4 configuration.
    
    Returns:
        dict: Device mapping for different tools
        
    Device allocation for 2xT4 (each 16GB):
        - GPU 0: LLaVA (4-bit ~4GB) + BiomedCLIP (~1GB)
        - GPU 1: Grounding DINO (~2GB) + MedSAM (~2GB)
    """
    n_gpus = get_gpu_count()
    
    if n_gpus >= 2:
        return {
            "llava": "cuda:0",
            "biomedclip": "cuda:0",
            "grounding_dino": "cuda:1",
            "medsam": "cuda:1",
        }
    elif n_gpus == 1:
        return {
            "llava": "cuda:0",
            "biomedclip": "cuda:0",
            "grounding_dino": "cuda:0",
            "medsam": "cuda:0",
        }
    else:
        return {
            "llava": "cpu",
            "biomedclip": "cpu",
            "grounding_dino": "cpu",
            "medsam": "cpu",
        }


# =============================================================================
# Weight Paths
# =============================================================================

def get_weights_dir():
    """Get weights directory based on environment."""
    if is_kaggle():
        return Path("/kaggle/working/weights")
    elif is_colab():
        return Path("/content/TriMedAgent/weights")
    else:
        return Path("weights")

def get_weight_paths():
    """
    Get model weight paths.
    
    Returns:
        dict: Paths to model weights
    """
    weights_dir = get_weights_dir()
    
    return {
        "grounding_dino": {
            "config": weights_dir / "GroundingDINO_SwinT_OGC.py",
            "checkpoint": weights_dir / "groundingdino_swint_ogc.pth",
        },
        "medsam": {
            "checkpoint": weights_dir / "medsam_vit_b.pth",
        },
    }


# =============================================================================
# Model URLs for Download
# =============================================================================

MODEL_URLS = {
    "grounding_dino": {
        "config": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "checkpoint": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    },
    "medsam": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
}


# =============================================================================
# HuggingFace Model Names
# =============================================================================

HF_MODELS = {
    "llava": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_PMC-ViT-B-16-224",
    "biomedclip_openclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "grounding_dino_hf": "IDEA-Research/grounding-dino-tiny",
}


# =============================================================================
# Default Configuration
# =============================================================================

class KaggleConfig:
    """Default configuration for Kaggle 2xT4 environment."""
    
    # Mode
    LITE_MODE = False  # False = Full FP16, True = 4-bit quantization
    
    # Tool loading
    LOAD_LLAVA = True
    LOAD_BIOMEDCLIP = True
    LOAD_DINO = True
    LOAD_MEDSAM = True
    LOAD_RAG = False  # RAG needs API key
    
    # LLaVA settings
    LLAVA_QUANTIZE_4BIT = True  # 4-bit quantization (recommended)
    LLAVA_MAX_TOKENS = 512
    
    # Detection thresholds
    DINO_BOX_THRESHOLD = 0.25
    DINO_TEXT_THRESHOLD = 0.25
    ZOOM_CONFIDENCE = 0.15
    GLOBAL_CONFIDENCE = 0.25
    STRICT_MAX_RATIO = 0.20  # Max 20% of image for nodules
    
    # ReAct loop
    MAX_ITERATIONS = 3
    ENABLE_VERIFICATION = True
    
    @classmethod
    def get_device_for_tool(cls, tool_name: str) -> str:
        """Get device for specific tool."""
        return get_device_map().get(tool_name, "cuda:0")


# =============================================================================
# Utility Functions
# =============================================================================

def download_weights():
    """Download all required weights."""
    import requests
    from tqdm import tqdm
    
    weights_dir = get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    paths = get_weight_paths()
    urls = MODEL_URLS
    
    def download_file(url, filepath):
        if filepath.exists():
            print(f"✅ Found {filepath.name}")
            return
        
        print(f"📥 Downloading {filepath.name}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                bar.update(len(data))
    
    # Download Grounding DINO
    download_file(urls["grounding_dino"]["config"], paths["grounding_dino"]["config"])
    download_file(urls["grounding_dino"]["checkpoint"], paths["grounding_dino"]["checkpoint"])
    
    # Download MedSAM
    download_file(urls["medsam"]["checkpoint"], paths["medsam"]["checkpoint"])
    
    print("✅ All weights ready!")


def print_gpu_info():
    """Print GPU information."""
    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return
    
    n_gpus = torch.cuda.device_count()
    print(f"✅ Detected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({mem_gb:.1f} GB)")
    
    device_map = get_device_map()
    print(f"\n📍 Device mapping:")
    for tool, device in device_map.items():
        print(f"   {tool}: {device}")
