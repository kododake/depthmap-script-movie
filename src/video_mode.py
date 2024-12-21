import pathlib
import traceback
from PIL import Image
import numpy as np
import os
import torch
from src import core
from src import backbone
from src.common_constants import GenerationOptions as go

def open_path_as_images(path, maybe_depthvideo=False, device='cpu'):
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
            # Suppose there are in fact 16 bits per pixel
            # If this is not the case, this is not a 16-bit depthvideo, so no need to process it this way
            gen = imageio_ffmpeg.read_frames(path, pix_fmt='gray16le', bits_per_pixel=16)
            video_info = next(gen)
            if video_info['pix_fmt'] == 'gray16le':
                width, height = video_info['size']
                frames = []
                for frame in gen:
                    # Not sure if this is implemented somewhere else
                    result = np.frombuffer(frame, dtype='uint16')
                    result.shape = (height, width)  # Why does it work? I don't remotely have any idea.
                    frames += [Image.fromarray(result)]
                    # TODO: Wrapping frames into Pillow objects is wasteful
                return video_info['fps'], frames
        finally:
            if 'gen' in locals():
                gen.close()
    if suffix.lower() in ['.webm', '.mp4', '.avi']:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(path)
        frames = []
        for frame in clip.iter_frames():
            img = torch.tensor(np.array(frame))
            frames.append(img.to(device))
            if torch.cuda.memory_allocated(device) > 20 * 1024 * 1024 * 1024:  # 20GB limit
                break
        return clip.fps, frames
    else:
        try:
            img = Image.open(path)
            img_tensor = torch.tensor(np.array(img)).to(device)
            return 1, [img_tensor]
        except Exception as e:
            raise Exception(f"Probably an unsupported file format: {suffix}") from e

def gen_video(video_path, outpath, inp, custom_depthmap=None, colorvids_bitrate=None, smoothening='none', device='cpu'):
    # Ensure all necessary keys are in the inp dictionary
    required_keys = [go.GEN_SIMPLE_MESH.name.lower(), go.GEN_INPAINTED_MESH.name.lower()]
    for key in required_keys:
        if key not in inp:
            inp[key] = False

    if inp[go.GEN_SIMPLE_MESH.name.lower()] or inp[go.GEN_INPAINTED_MESH.name.lower()]:
        return 'Creating mesh-videos is not supported. Please split video into frames and use batch processing.'

    fps, input_images = open_path_as_images(os.path.abspath(video_path), device=device)
    os.makedirs(backbone.get_outpath(), exist_ok=True)

    if custom_depthmap is None:
        print('Generating depthmaps for the video frames')
        needed_keys = [go.COMPUTE_DEVICE, go.MODEL_TYPE, go.BOOST, go.NET_SIZE_MATCH, go.NET_WIDTH, go.NET_HEIGHT]
        needed_keys = [x.name.lower() for x in needed_keys]
        first_pass_inp = {k: v for (k, v) in inp.items() if k in needed_keys}
        # We need predictions where frames are not normalized separately.
        first_pass_inp[go.DO_OUTPUT_DEPTH_PREDICTION] = True
        # No need in normalized frames. Properly processed depth video will be created in the second pass
        first_pass_inp[go.DO_OUTPUT_DEPTH.name] = False

        gen_obj = core.core_generation_funnel(None, input_images, None, None, first_pass_inp)
        input_depths = [x[2] for x in list(gen_obj)]
        input_depths = process_predicitons(input_depths, smoothening)
    else:
        print('Using custom depthmap video')
        cdm_fps, input_depths = open_path_as_images(os.path.abspath(custom_depthmap), maybe_depthvideo=True, device=device)
        assert len(input_depths) == len(input_images), 'Custom depthmap video length does not match input video length'
        if input_depths[0].size != input_images[0].size:
            print('Warning! Input video size and depthmap video size are not the same!')

    print('Generating output frames')
    img_results = list(core.core_generation_funnel(None, input_images, input_depths, None, inp))
    gens = list(set(map(lambda x: x[1], img_results)))

    print('Saving generated frames as video outputs')
    for gen in gens:
        if gen == 'depth' and custom_depthmap is not None:
            continue

        imgs = [x[2] for x in img_results if x[1] == gen]
        basename = f'{gen}_video'
        frames_to_video(fps, imgs, outpath, f"depthmap-{backbone.get_next_sequence_number(outpath, basename)}-{basename}", colorvids_bitrate)

    print('Generating stereo images for each frame')
    stereo_images = []
    for image, depth_map in zip(input_images, input_depths):
        stereo_image = create_stereoimages(image, depth_map, inp[go.STEREO_DIVERGENCE], inp[go.STEREO_SEPARATION], inp[go.STEREO_MODES], inp[go.STEREO_BALANCE], inp[go.STEREO_OFFSET_EXPONENT], inp[go.STEREO_FILL_ALGO])
        stereo_images.append(stereo_image[0])
    
    frames_to_video(fps, stereo_images, outpath, 'stereo_video')

    print('All done. Video(s) saved!')
    return '<h3>Videos generated</h3>' if len(gens) > 1 else '<h3>Video generated</h3>' if len(gens) is 1 else '<h3>Nothing generated - please check the settings and try again</h3>'

from src.stereoimage_generation import create_stereoimages



def process_video_with_stereo(video_path, output_path, divergence=2.0, separation=0.5, modes=['left-right'], stereo_balance=0.0, stereo_offset_exponent=1.0, fill_technique='polylines_sharp'):
    # Extract frames from video
    fps, frames = open_path_as_images(video_path)
    
    # Create stereo images for each frame
    stereo_frames = []
    for frame in frames:
        depth_map = generate_depth_map(frame)  # Assuming you have a function to generate depth map for each frame
        stereo_image = create_stereoimages(frame, depth_map, divergence, separation, modes, stereo_balance, stereo_offset_exponent, fill_technique)
        stereo_frames.append(stereo_image[0])  # Assuming modes has at least one mode
    
    # Combine processed frames back into a video
    frames_to_video(fps, stereo_frames, output_path, 'stereo_video')

def frames_to_video(fps, frames, path, name, colorvids_bitrate=None):
    if frames[0].mode == 'I;16':  # depthmap video
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
            except:
                traceback.print_exc()
        if not done:
            raise Exception('Saving the video failed!')


def process_predicitons(predictions, smoothening='none'):
    def global_scaling(objs, a=None, b=None):
        """Normalizes objs, but uses (a, b) instead of (minimum, maximum) value of objs, if supplied"""
        normalized = []
        min_value = a if a is not None else min([obj.min() for obj in objs])
        max_value = b if b is not None else max([obj.max() for obj in objs])
        for obj in objs:
            normalized += [(obj - min_value) / (max_value - min_value)]
        return normalized

    print('Processing generated depthmaps')
    # TODO: Detect cuts and process segments separately
    if smoothening == 'none':
        return global_scaling(predictions)
    elif smoothening == 'experimental':
        processed = []
        clip = lambda val: min(max(0, val), len(predictions) - 1)
        for i in range(len(predictions)):
            f = np.zeros_like(predictions[i])
            for u, mul in enumerate([0.10, 0.20, 0.40, 0.20, 0.10]):  # Eyeballed it, math person please fix this
                f += mul * predictions[clip(i + (u - 2))]
            processed += [f]
        # This could have been deterministic monte carlo... Oh well, this version is faster.
        a, b = np.percentile(np.stack(processed), [0.5, 99.5])
        return global_scaling(predictions, a, b)
    return predictions


def gen_video(video_path, outpath, inp, custom_depthmap=None, colorvids_bitrate=None, smoothening='none'):
    # Ensure all necessary keys are in the inp dictionary
    required_keys = [go.GEN_SIMPLE_MESH.name.lower(), go.GEN_INPAINTED_MESH.name.lower()]
    for key in required_keys:
        if key not in inp:
            inp[key] = False

    if inp[go.GEN_SIMPLE_MESH.name.lower()] or inp[go.GEN_INPAINTED_MESH.name.lower()]:
        return 'Creating mesh-videos is not supported. Please split video into frames and use batch processing.'

    fps, input_images = open_path_as_images(os.path.abspath(video_path))
    os.makedirs(backbone.get_outpath(), exist_ok=True)

    if custom_depthmap is None:
        print('Generating depthmaps for the video frames')
        needed_keys = [go.COMPUTE_DEVICE, go.MODEL_TYPE, go.BOOST, go.NET_SIZE_MATCH, go.NET_WIDTH, go.NET_HEIGHT]
        needed_keys = [x.name.lower() for x in needed_keys]
        first_pass_inp = {k: v for (k, v) in inp.items() if k in needed_keys}
        # We need predictions where frames are not normalized separately.
        first_pass_inp[go.DO_OUTPUT_DEPTH_PREDICTION] = True
        # No need in normalized frames. Properly processed depth video will be created in the second pass
        first_pass_inp[go.DO_OUTPUT_DEPTH.name] = False

        gen_obj = core.core_generation_funnel(None, input_images, None, None, first_pass_inp)
        input_depths = [x[2] for x in list(gen_obj)]
        input_depths = process_predicitons(input_depths, smoothening)
    else:
        print('Using custom depthmap video')
        cdm_fps, input_depths = open_path_as_images(os.path.abspath(custom_depthmap), maybe_depthvideo=True)
        assert len(input_depths) == len(input_images), 'Custom depthmap video length does not match input video length'
        if input_depths[0].size != input_images[0].size:
            print('Warning! Input video size and depthmap video size are not the same!')

    print('Generating output frames')
    img_results = list(core.core_generation_funnel(None, input_images, input_depths, None, inp))
    gens = list(set(map(lambda x: x[1], img_results)))

    print('Saving generated frames as video outputs')
    for gen in gens:
        if gen == 'depth' and custom_depthmap is not None:
            continue

        imgs = [x[2] for x in img_results if x[1] == gen]
        basename = f'{gen}_video'
        frames_to_video(fps, imgs, outpath, f"depthmap-{backbone.get_next_sequence_number(outpath, basename)}-{basename}",
                        colorvids_bitrate)

    print('Generating stereo images for each frame')
    stereo_images = []
    for image, depth_map in zip(input_images, input_depths):
        stereo_image = create_stereoimages(image, depth_map, inp[go.STEREO_DIVERGENCE], inp[go.STEREO_SEPARATION], inp[go.STEREO_MODES], inp[go.STEREO_BALANCE], inp[go.STEREO_OFFSET_EXPONENT], inp[go.STEREO_FILL_ALGO])
        stereo_images.append(stereo_image[0])
    
    frames_to_video(fps, stereo_images, outpath, 'stereo_video')

    print('All done. Video(s) saved!')
    return '<h3>Videos generated</h3>' if len(gens) > 1 else '<h3>Video generated</h3>' if len(gens) == 1 \
        else '<h3>Nothing generated - please check the settings and try again</h3>'
