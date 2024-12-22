import pathlib
import traceback
from PIL import Image
import numpy as np
import os
import torch
from src import core
from src import backbone
from src.common_constants import GenerationOptions as go
from config import MEMORY_LIMIT_GB  # Import from config.py
import concurrent.futures
from src.stereoimage_generation import create_stereoimages

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set memory limit based on the global variable
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_GB * 10**9 / torch.cuda.get_device_properties(0).total_memory)

from moviepy.video.io.VideoFileClip import VideoFileClip

def open_path_as_images(path, maybe_depthvideo=False, device=device, batch_size=10):
    """Takes the filepath, returns (fps, frames). Every frame is a Pillow Image object"""
    suffix = pathlib.Path(path).suffix
    if suffix.lower() == '.gif':
        frames = []
        img = Image.open(path)
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.convert('RGB'))
        return 1000 / img.info['duration'], frames
    if suffix.lower() == '.mts':
        import imageio_ffmpeg
        import av
        container = av.open(path)
        frames = []
        for packet in container.demux(video=0):
            for frame in packet.decode():
                # Convert the frame to a NumPy array
                numpy_frame = frame.to_ndarray(format='rgb24')
                # Convert the NumPy array to a Pillow Image
                image = Image.fromarray(numpy_frame)
                frames.append(image)
        fps = float(container.streams.video[0].average_rate)
        container.close()
        return fps, frames
    if suffix.lower() in ['.avi'] and maybe_depthvideo:
        try:
            import imageio_ffmpeg
            gen = imageio_ffmpeg.read_frames(path, pix_fmt='gray16le', bits_per_pixel=16)
            video_info = next(gen)
            if video_info['pix_fmt'] == 'gray16le':
                width, height = video_info['size']
                frames = []
                for frame in gen:
                    result = np.frombuffer(frame, dtype='uint16')
                    result.shape = (height, width)
                    frames += [Image.fromarray(result)]
                return video_info['fps'], frames
        finally:
            if 'gen' in locals():
                gen.close()
    if suffix.lower() in ['.webm', '.mp4', '.avi']:
        clip = VideoFileClip(path)
        fps = clip.fps
        for start_frame in range(0, int(clip.fps * clip.duration), batch_size):
            frames = []
            for frame in clip.iter_frames():
                img = torch.tensor(np.array(frame)).to(device)
                frames.append(img)
                if len(frames) == batch_size:
                    yield fps, frames
                    frames = []
            if frames:
                yield fps, frames
        return
    else:
        try:
            img = Image.open(path)
            img_tensor = torch.tensor(np.array(img)).to(device)
            return 1, [img_tensor]
        except Exception as e:
            raise Exception(f"Probably an unsupported file format: {suffix}") from e

def gen_video(video_path, outpath, inp, custom_depthmap=None, colorvids_bitrate=None, smoothening='none', device=device):
    # Ensure all necessary keys are in the inp dictionary
    required_keys = [go.GEN_SIMPLE_MESH.name.lower(), go.GEN_INPAINTED_MESH.name.lower()]
    for key in required_keys:
        if key not in inp:
            inp[key] = False

    if inp[go.GEN_SIMPLE_MESH.name.lower()] or inp[go.GEN_INPAINTED_MESH.name.lower()]:
        return 'Creating mesh-videos is not supported. Please split video into frames and use batch processing.'

    os.makedirs(backbone.get_outpath(), exist_ok=True)

    if custom_depthmap is None:
        print('Generating depthmaps for the video frames')
        needed_keys = [go.COMPUTE_DEVICE, go.MODEL_TYPE, go.BOOST, go.NET_SIZE_MATCH, go.NET_WIDTH, go.NET_HEIGHT]
        needed_keys = [x.name.lower() for x in needed_keys]
        first_pass_inp = {k: v for (k, v) in inp.items() if k in needed_keys}
        first_pass_inp[go.DO_OUTPUT_DEPTH_PREDICTION] = True
        first_pass_inp[go.DO_OUTPUT_DEPTH.name] = False

        for fps, input_images in open_path_as_images(os.path.abspath(video_path), device=device):
            gen_obj = core.core_generation_funnel(None, input_images, None, None, first_pass_inp)
            input_depths = [x[2] for x in list(gen_obj)]
            input_depths = process_predictions(input_depths, smoothening)
            process_and_save(input_images, input_depths, fps, outpath, inp, colorvids_bitrate, custom_depthmap)
    else:
        print('Using custom depthmap video')
        for fps, input_images in open_path_as_images(os.path.abspath(video_path), device=device):
            cdm_fps, input_depths = open_path_as_images(os.path.abspath(custom_depthmap), maybe_depthvideo=True, device=device)
            assert len(input_depths) == len(input_images), 'Custom depthmap video length does not match input video length'
            if input_depths[0].size != input_images[0].size:
                print('Warning! Input video size and depthmap video size are not the same!')
            process_and_save(input_images, input_depths, fps, outpath, inp, colorvids_bitrate)

    print('All done. Video(s) saved!')
    gens = [video for video in os.listdir(outpath) if video.endswith(('.avi', '.mp4', '.webm'))]
    return '<h3>Videos generated</h3>' if len(gens) > 1 else '<h3>Video generated</h3>' if len(gens) == 1 else '<h3>Nothing generated - please check the settings and try again</h3>'

def process_and_save(input_images, input_depths, fps, outpath, inp, colorvids_bitrate, custom_depthmap=None):
    print('Generating output frames')
    img_results = list(core.core_generation_funnel(None, input_images, input_depths, None, inp))
    gens = list(set(map(lambda x: x[1], img_results)))

    if not gens:
        raise ValueError("No generated frames found, please check the settings and try again.")

    print('Saving generated frames as video outputs')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for gen in gens:
            if gen == 'depth' and custom_depthmap is not None:
                continue

            # Convert Image objects to tensors and move to GPU
            try:
                imgs = [torch.tensor(np.array(x[2])).to(device) if isinstance(x[2], Image.Image) else torch.tensor(x[2]).to(device) for x in img_results if x[1] == gen]
            except IndexError as e:
                print(f"IndexError: {e} - Check if the generated frames are correctly processed.")
                continue
            basename = f'{gen}_video'
            futures.append(executor.submit(frames_to_video, fps, imgs, outpath, f"depthmap-{backbone.get_next_sequence_number(outpath, basename)}-{basename}", colorvids_bitrate))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error saving video: {e}")

    # 必要なキーを設定
    required_keys = [
        go.STEREO_DIVERGENCE, go.STEREO_SEPARATION, go.STEREO_MODES,
        go.STEREO_BALANCE, go.STEREO_OFFSET_EXPONENT, go.STEREO_FILL_ALGO
    ]
    for key in required_keys:
        inp.setdefault(key.name.lower(), key.df)

    print('Generating stereo images for each frame')
    stereo_images = []
    for image, depth_map in zip(input_images, input_depths):
        stereo_image = create_stereoimages(
            torch.tensor(np.array(image)).to(device) if isinstance(image, Image.Image) else torch.tensor(image).to(device),
            torch.tensor(np.array(depth_map)).to(device) if isinstance(depth_map, Image.Image) else torch.tensor(depth_map).to(device),
            inp[go.STEREO_DIVERGENCE.name.lower()], inp[go.STEREO_SEPARATION.name.lower()],
            inp[go.STEREO_MODES.name.lower()], inp[go.STEREO_BALANCE.name.lower()],
            inp[go.STEREO_OFFSET_EXPONENT.name.lower()], inp[go.STEREO_FILL_ALGO.name.lower()]
        )
        stereo_image_np = np.array(stereo_image[0])  # Convert to NumPy array
        stereo_images.append(Image.fromarray(stereo_image_np))  # Convert back to PIL Image

    frames_to_video(fps, stereo_images, outpath, 'stereo_video')

def process_video_with_stereo(video_path, output_path, divergence=2.0, separation=0.5, modes=['left-right'], stereo_balance=0.0, stereo_offset_exponent=1.0, fill_technique='polylines_sharp'):
    fps, frames = open_path_as_images(video_path, device=device)
    stereo_frames = []
    for frame in frames:
        depth_map = generate_depth_map(frame)
        stereo_image = create_stereoimages(frame, depth_map, divergence, separation, modes, stereo_balance, stereo_offset_exponent, fill_technique)
        stereo_frames.append(stereo_image[0])
    frames_to_video(fps, stereo_frames, output_path, 'stereo_video')

def frames_to_video(fps, frames, path, name, colorvids_bitrate=None):
    if not frames:
        raise ValueError("No frames available to process")

    try:
        # Ensure frames are in the correct format
        if isinstance(frames[0], torch.Tensor):
            frames = [Image.fromarray(frame.cpu().numpy()) for frame in frames]
        
        if frames[0].mode == 'I;16':
            import imageio_ffmpeg
            writer = imageio_ffmpeg.write_frames(
                os.path.join(path, f"{name}.avi"), frames[0].size, 'gray16le', 'gray16le', fps, codec='ffv1',
                macro_block_size=1)
            try:
                writer.send(None)
                for frame in frames:
                    writer.send(np.array(frame))
            finally:
                writer.close()
        else:
            arrs = [np.asarray(frame) for frame in frames]
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            clip = ImageSequenceClip(arrs, fps=fps)
            done = False
            priority = [('avi', 'png'), ('avi', 'rawvideo'), ('mp4', 'libx264'), ('webm', 'libvpx')]
            if colorvids_bitrate:
                priority = reversed(priority)
            for v_format, codec in priority:
                try:
                    br = f'{colorvids_bitrate}k' if codec not in ['png', 'rawvideo'] else None
                    clip.write_videofile(os.path.join(path, f"{name}.{v_format}"), codec=codec, bitrate=br)
                    done = True
                    break
                except Exception as e:
                    print(f"Exception: {e} - Failed to save video in format {v_format} with codec {codec}")
                    traceback.print_exc()
            if not done:
                raise Exception('Saving the video failed!')
    except Exception as e:
        print(f"Error in frames_to_video: {e}")
        raise

def process_predictions(predictions, smoothening='none'):
    def global_scaling(objs, a=None, b=None):
        normalized = []
        min_value = a if a is not None else min([obj.min() for obj in objs])
        max_value = b if b is not None else max([obj.max() for obj in objs])
        for obj in objs:
            normalized += [(obj - min_value) / (max_value - min_value)]
        return normalized

    print('Processing generated depthmaps')
    if smoothening == 'none':
        return global_scaling(predictions)
    elif smoothening == 'experimental':
        processed = []
        clip = lambda val: min(max(0, val), len(predictions) - 1)
        for i in range(len(predictions)):
            f = np.zeros_like(predictions[i])
            for u, mul in enumerate([0.10, 0.20, 0.40, 0.20, 0.10]):
                f += mul * predictions[clip(i + (u - 2))]
            processed += [f]
        a, b = np.percentile(np.stack(processed), [0.5, 99.5])
        return global_scaling(predictions, a, b)
    return predictions
