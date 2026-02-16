from pathlib import Path
import os
import torch

from mast3r.demo_glomap import get_args_parser, reconstruct_glomap
from kapture.converter.colmap.database_extra import kapture_to_colmap

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)
    
    image_dir = Path("path/to/your/images")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    filelist = []
    if image_dir.exists():
        for file_path in image_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                filelist.append(str(file_path))

    filelist.sort()
    
    os.makedirs("output", exist_ok=True)
    
    retrieval_model = getattr(args, 'retrieval_model', None)
    scenegraph_type = getattr(args, 'scenegraph_type', 'swin')
    winsize = getattr(args, 'winsize', 1)
    win_cyclic = getattr(args, 'win_cyclic', False)
    refid = getattr(args, 'refid', None)
    shared_intrinsics = getattr(args, 'shared_intrinsics', False)
    
    reconstruct_glomap(args.glomap_bin, "output", model, retrieval_model,
        args.device, args.silent, args.image_size, filelist, scenegraph_type,
        winsize, win_cyclic, refid, shared_intrinsics)