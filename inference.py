from pathlib import Path
import os
import torch
import argparse

from mast3r.demo_glomap import reconstruct_glomap
from kapture.converter.colmap.database_extra import kapture_to_colmap

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

from dust3r.demo import set_print_with_timestamp

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def setup_args():
    parser = argparse.ArgumentParser(description="MASt3R Reconstruction with GLOMAP")

    # --- Required Arguments ---
    parser.add_argument('--images_dir', type=Path, required=True, help="Path to input images folder")
    parser.add_argument('--glomap_bin', type=str, required=True, help="Path to glomap executable binary")

    # --- Model Arguments ---
    parser.add_argument('--model_name', type=str, default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", 
                        help="Name of the model to load from HuggingFace")
    parser.add_argument('--weights', type=str, default=None, help="Path to local weights (overrides model_name)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Processing Arguments ---
    parser.add_argument('--image_size', type=int, default=512, help="Long edge resize size")
    parser.add_argument('--retrieval_model', type=str, default=None, help="Retrieval model (e.g., sentinel, netvlad)")
    parser.add_argument('--scenegraph_type', type=str, default="swin", help="Type of scenegraph (complete, swin, retrieval)")
    
    # --- Reconstruction Tuning ---
    parser.add_argument('--winsize', type=int, default=1)
    parser.add_argument('--win_cyclic', action='store_true', help="Use cyclic windowing")
    parser.add_argument('--refid', type=int, default=0, help="Reference image index")
    parser.add_argument('--shared_intrinsics', action='store_true', help="Assume all images share the same camera")
    parser.add_argument('--silent', action='store_true', help="Disable verbose output")

    return parser.parse_args()

def get_filelist(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.exists():
        raise ValueError(f"Image directory {image_dir} does not exist.")
    
    filelist = [str(f) for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    filelist.sort()
    
    if not filelist:
        raise ValueError(f"No valid images found in {image_dir}")
    return filelist

if __name__ == '__main__':
    args = setup_args()
    set_print_with_timestamp()

    weights_path = args.weights if args.weights else f"naver/{args.model_name}"

    print(f">> Loading model: {weights_path} on {args.device}")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)
    
    filelist = get_filelist(args.images_dir)
    os.makedirs("output", exist_ok=True)
    
    print(f">> Starting GLOMAP reconstruction with {len(filelist)} images...")
    
    reconstruct_glomap(
        args.glomap_bin, 
        "output", 
        model, 
        args.retrieval_model,
        args.device, 
        args.silent, 
        args.image_size, 
        filelist, 
        args.scenegraph_type,
        args.winsize, 
        args.win_cyclic, 
        args.refid, 
        args.shared_intrinsics
    )