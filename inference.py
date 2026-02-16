import functools
import pycolmap
import os
import torch
import tempfile
from contextlib import nullcontext
import gradio

from mast3r.demo_glomap import get_args_parser

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def main_demo(glomap_bin, tmpdirname, model, retrieval_model, device, image_size, server_name, server_port,
              silent=False, share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, glomap_bin, tmpdirname, gradio_delete_cache, model,
                                  retrieval_model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                    info="Only optimize one set of intrinsics for all views")
                scenegraph_type = gradio.Dropdown(available_scenegraph_type,
                                                  value='complete', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                with gradio.Column(visible=False) as win_col:
                    winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                            minimum=1, maximum=1, step=1)
                    win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                      minimum=0, maximum=0, step=1, visible=False)
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.01, minimum=0.001, maximum=1.0, step=0.001)
            with gradio.Row():
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[win_col, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, shared_intrinsics],
                          outputs=[scene, outmodel])
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, transparent_cams, cam_size],
                            outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, transparent_cams, cam_size],
                                    outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)


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

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)
    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        main_demo(args.glomap_bin, cache_path, model, args.retrieval_model, args.device, args.image_size, 
            silent=args.silent, share=args.share, gradio_delete_cache=args.gradio_delete_cache)
