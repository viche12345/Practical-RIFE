import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
import shutil
import subprocess

warnings.filterwarnings("ignore")

def create_video_from_pngs(png_dir, output_path, fps, audio_source=None):
    """
    Create MP4 video from PNG sequence using FFmpeg with custom settings
    """
    # Build FFmpeg command for PNG sequence to MP4
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-framerate', str(fps),
        '-i', os.path.join(png_dir, '%07d.png'),
        '-c:v', 'libx264',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-r', str(fps)  # Output framerate
    ]
    
    # If audio source is provided, add audio stream
    if audio_source:
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        cmd.append(temp_video)
        
        # First create video without audio
        print(f"Creating video from PNG sequence...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
        # Then add audio
        audio_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_source,
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'copy',  # Copy audio stream without re-encoding
            '-shortest',     # Match shortest stream duration
            output_path
        ]
        
        print(f"Adding audio to video...")
        result = subprocess.run(audio_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Audio merge failed: {result.stderr}")
            # If audio merge fails, rename temp video as final output
            os.rename(temp_video, output_path)
            print("Video created without audio")
        else:
            # Remove temp video file
            os.remove(temp_video)
            print("Video created with audio")
    else:
        cmd.append(output_path)
        print(f"Creating video from PNG sequence...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        print("Video created successfully")
    
    return True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--keep-pngs', dest='keep_pngs', action='store_true', help='keep PNG files after creating video')
parser.add_argument('--png-only', dest='png_only', action='store_true', help='only output PNG sequence, skip video creation')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if not args.png_only and fpsNotAssigned:
        print("Audio will be merged after video creation")
    else:
        print("Will not merge audio (PNG-only mode or custom FPS)")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]

h, w, _ = lastframe.shape

# Always create PNG output directory
png_output_dir = 'vid_out'
if not os.path.exists(png_output_dir):
    os.makedirs(png_output_dir)
else:
    # Clear existing PNG files
    for f in os.listdir(png_output_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(png_output_dir, f))

# Determine final video output path
if not args.png_only:
    if args.output is not None:
        final_video_path = args.output
    else:
        if not args.video is None:
            final_video_path = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
        else:
            final_video_path = 'interpolated_{}X_{}fps.{}'.format(args.multi, int(np.round(args.fps)), args.ext)

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        # Always write PNG files
        cv2.imwrite(os.path.join(png_output_dir, '{:0>7d}.png'.format(cnt)), item[:, :, ::-1])
        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):    
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None # save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get() # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, scale=args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
    if ssim < 0.2:
        output = []
        for i in range(args.multi - 1):
            output.append(I0)
    else:
        output = make_inference(I0, I1, args.multi - 1)

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()

print(f"PNG sequence saved to {png_output_dir}/")

# Create video from PNG sequence using FFmpeg
if not args.png_only:
    audio_source = args.video if (not args.video is None and fpsNotAssigned) else None
    success = create_video_from_pngs(png_output_dir, final_video_path, args.fps, audio_source)
    
    if success:
        print(f"Video saved to {final_video_path}")
        
        # Clean up PNG files if not requested to keep them
        if not args.keep_pngs:
            print("Cleaning up PNG files...")
            shutil.rmtree(png_output_dir)
            print("PNG files removed")
    else:
        print("Video creation failed. PNG files preserved.")
else:
    print("PNG-only mode: Video creation skipped")