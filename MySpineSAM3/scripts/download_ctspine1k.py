#!/usr/bin/env python
"""
Download CTSpine1K Volumes from HuggingFace
============================================
Downloads ONLY the volumes/COLONOG directory (CT images) from HuggingFace.
Segmentation masks are much smaller and already downloaded.

Usage:
    # Download only volumes
    python scripts/download_ctspine1k.py --output ./data/ctspine1k_raw --volumes-only
    
    # Download only segmentations  
    python scripts/download_ctspine1k.py --output ./data/ctspine1k_raw --seg-only
    
    # Download everything
    python scripts/download_ctspine1k.py --output ./data/ctspine1k_raw
"""

import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download CTSpine1K data")
    parser.add_argument("--output", default="./data/ctspine1k_raw", help="Output directory")
    parser.add_argument("--volumes-only", action="store_true", help="Download only volumes/COLONOG")
    parser.add_argument("--seg-only", action="store_true", help="Download only segmentation masks")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")
    
    # Set up patterns based on what to download
    if args.volumes_only:
        allow_patterns = ["raw_data/volumes/COLONOG/*"]
        print("Downloading ONLY volumes/COLONOG (~150GB)...")
    elif args.seg_only:
        allow_patterns = ["raw_data/segmentation/*"]
        print("Downloading ONLY segmentation masks (~300MB)...")
    else:
        allow_patterns = None
        print("Downloading ALL CTSpine1K data (~150GB)...")
    
    print(f"Output directory: {output_dir.absolute()}")
    print("This may take a while... (supports resume)")
    
    snapshot_download(
        repo_id="alexanderdann/CTSpine1K",
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )
    
    print(f"\nDownload complete! Data saved to: {output_dir}")
    print("\nTo use in training, set in config:")
    print("  data:")
    print("    source: 'local'")
    print(f"    local_data_dir: '{output_dir}'")


if __name__ == "__main__":
    main()
