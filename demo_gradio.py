#!/usr/bin/env python3
import functools
import gc
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from src.utils.image import process_image


# 1) Core model inference
def get_reconstructed_scene(outdir, model, device):
    # Load Images
    image_files = sorted(
        [
            os.path.join(outdir, "images", f)
            for f in os.listdir(os.path.join(outdir, "images"))
        ]
    )
    images = [process_image(img_path) for img_path in image_files]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, c, h, w = images.shape

    assert c == 3, "Images must have 3 channels"

    # Run Inference
    gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    # Save the results
    pred_all_extrinsic = pred_context_pose["extrinsic"]
    pred_all_intrinsic = pred_context_pose["intrinsic"]
    video, depth_colored = save_interpolated_video(
        pred_all_extrinsic,
        pred_all_intrinsic,
        b,
        h,
        w,
        gaussians,
        outdir,
        model.decoder,
    )

    plyfile = os.path.join(outdir, "gaussians.ply")
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(plyfile),
        save_sh_dc_only=True,
    )

    # Clean up
    torch.cuda.empty_cache()
    return plyfile, video, depth_colored


# 2) Handle uploaded video/images --> produce target_dir + images
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(
                    target_dir_images, f"{video_frame_num:06}.png"
                )
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(
        f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds"
    )
    return target_dir, image_paths


# 3) Update gallery on upload
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths


# 4) Reconstruction: uses the target_dir plus any viz parameters
def gradio_demo(
    target_dir,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = (
        sorted(os.listdir(target_dir_images))
        if os.path.isdir(target_dir_images)
        else []
    )
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]

    print("Running run_model...")
    import traceback
    try:
        with torch.no_grad():
            plyfile, video, depth_colored = get_reconstructed_scene(
                target_dir, model, device
            )
    except Exception as e:
        error_msg = f"Error during reconstruction: {str(e)}\n{traceback.format_exc()}"
        try:
            print(error_msg, flush=True)
        except (OSError, UnicodeEncodeError):
            # Fallback for Windows encoding issues (errno 22)
            sys.stderr.write(error_msg + "\n")
            sys.stderr.flush()
        # Also write to a log file
        with open("reconstruction_errors.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{error_msg}\n")
        raise

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")

    return plyfile, video, depth_colored


def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None, None, None


if __name__ == "__main__":
    server_name = "127.0.0.1"
    server_port = None
    share = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AnySplat.from_pretrained(
        "lhjiang/anysplat"
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )
    css = ""
    with gr.Blocks(css=css, title="Evolver", theme=theme) as demo:
        gr.Markdown(
            """
            <h1 style='text-align: center;'>Evolver</h1>
            <p style='text-align: center; color: #666;'>3D Gaussian Splatting from Images</p>
            """
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Input Data"):
                        input_video = gr.Video(label="Upload Video", interactive=True)
                        input_images = gr.File(
                            file_count="multiple",
                            label="Upload Images",
                            interactive=True,
                        )

                        image_gallery = gr.Gallery(
                            label="Preview",
                            columns=4,
                            height="300px",
                            object_fit="contain",
                            preview=True,
                        )

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Output"):
                        with gr.Column():
                            reconstruction_output = gr.Model3D(
                                label="3D Reconstructed Gaussian Splat",
                                height=540,
                                zoom_speed=0.5,
                                pan_speed=0.5,
                                camera_position=[20, 20, 20],
                            )

                        with gr.Row():
                            with gr.Row():
                                rgb_video = gr.Video(
                                    label="RGB Video", interactive=False, autoplay=True
                                )
                                depth_video = gr.Video(
                                    label="Depth Video",
                                    interactive=False,
                                    autoplay=True,
                                )

                        with gr.Row():
                            submit_btn = gr.Button(
                                "Reconstruct", scale=1, variant="primary"
                            )
                            clear_btn = gr.ClearButton(
                                [
                                    input_video,
                                    input_images,
                                    reconstruction_output,
                                    target_dir_output,
                                    image_gallery,
                                    rgb_video,
                                    depth_video,
                                ],
                                scale=1,
                            )


        submit_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[reconstruction_output, rgb_video, depth_video],
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
            ],
            outputs=[reconstruction_output, rgb_video, depth_video],
        )

        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery],
        )

        # demo.launch(share=share, server_name=server_name, server_port=server_port)
        import sys
        import traceback
        import atexit
        
        # Save original stderr
        original_stderr = sys.stderr
        
        # Create a class that writes to both file and original stderr
        class TeeStderr:
            def __init__(self, file, original):
                self.file = file
                self.original = original
            
            def write(self, text):
                self.file.write(text)
                self.file.flush()
                self.original.write(text)
                self.original.flush()
            
            def flush(self):
                self.file.flush()
                self.original.flush()
            
            def close(self):
                self.file.close()
        
        # Open log file and create tee wrapper
        log_file = open("runtime_errors.log", "w", encoding="utf-8")
        tee_stderr = TeeStderr(log_file, original_stderr)
        sys.stderr = tee_stderr
        
        # Register cleanup function to restore stderr and close file on exit
        def cleanup():
            sys.stderr = original_stderr
            if not log_file.closed:
                log_file.close()
        
        atexit.register(cleanup)
        
        print("Starting Gradio app...", flush=True)
        print("Errors will be logged to runtime_errors.log", flush=True)
        try:
            demo.queue(max_size=20).launch(show_error=True, share=False)
        except Exception as e:
            traceback.print_exc()
            cleanup()
            raise
