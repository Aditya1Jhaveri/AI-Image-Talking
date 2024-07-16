from config import *
import cv2
import glob
import numpy as np
import os
from basicsr.utils import imwrite
from pathos.pools import ParallelPool
import subprocess
import platform
from mutagen.wave import WAVE
import tqdm
from p_tqdm import *
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
# from RealESRGAN import RealESRGAN
from gfpgan import GFPGANer
from tqdm import tqdm

def vid2frames(vidPath, framesOutPath):
    print(vidPath)
    print(framesOutPath)
    vidcap = cv2.VideoCapture(vidPath)
    success,image = vidcap.read()
    frame = 1
    while success:
      cv2.imwrite(os.path.join(framesOutPath, str(frame).zfill(5) + '.png'), image)
      success,image = vidcap.read()
      frame += 1

def restore_frames(audiofilePath, videoOutPath, improveOutputPath):
    no_of_frames = count_files(improveOutputPath)
    audio_duration = get_audio_duration(audiofilePath)
    framesPath = improveOutputPath + "/%5d.png"
    fps = no_of_frames/audio_duration
    command = f"ffmpeg -y -r {fps} -f image2 -i {framesPath} -i {audiofilePath} -vcodec mpeg4 -b:v 20000k {videoOutPath}"
    print(command)
    subprocess.call(command, shell=platform.system() != 'Windows')

def get_audio_duration(audioPath):
    audio = WAVE(audioPath)
    duration = audio.info.length
    return duration    

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


############# GFPGAN Model 
def process(img_path, improveOutputPath):
    only_center_face = True
    aligned = True
    weight = 0.5
    upscale = 1
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.4'
    url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

   # Determine model paths
    model_path = os.path.join('gfpgan_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # Download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None)

    # Read image
    img_name = os.path.basename(img_path)
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=aligned,
        only_center_face=only_center_face,
        paste_back=True,
        weight=weight)

    # Save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # Save cropped face
        save_crop_path = os.path.join(improveOutputPath, 'cropped_faces', f'{basename}_{idx}.png')
        cv2.imwrite(save_crop_path, cropped_face)
        # Save restored face
        save_restore_path = os.path.join(improveOutputPath, 'restored_faces', f'{basename}_{idx}.png')
        cv2.imwrite(save_restore_path, restored_face)
        # Save comparison image
        cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
        cv2.imwrite(os.path.join(improveOutputPath, 'cmp', f'{basename}_{idx}.png'), cmp_img)

    # Save restored img
    if restored_img is not None:
        extension = ext[1:] if ext else 'png'
        save_restore_path = os.path.join(improveOutputPath, 'restored_imgs', f'{basename}.{extension}')
        cv2.imwrite(save_restore_path, restored_img)

def improve(improveInputPath, improveOutputPath):
    if improveInputPath.endswith('/'):
        improveInputPath = improveInputPath[:-1]
    if os.path.isfile(improveInputPath):
        img_list = [improveInputPath]
    else:
        img_list = sorted(glob.glob(os.path.join(improveInputPath, '*')))

    os.makedirs(os.path.join(improveOutputPath, 'cropped_faces'), exist_ok=True)
    os.makedirs(os.path.join(improveOutputPath, 'restored_faces'), exist_ok=True)
    os.makedirs(os.path.join(improveOutputPath, 'cmp'), exist_ok=True)
    os.makedirs(os.path.join(improveOutputPath, 'restored_imgs'), exist_ok=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(lambda img: process(img, improveOutputPath), img_list), total=len(img_list), desc="Processing images"))

    print("All images processed.")
    
######### REAL-ESRGAN MODEL     

# def improve(disassembledPath, improvedPath):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = RealESRGAN(device, scale=4)
#     model.load_weights('weights/RealESRGAN_x4.pth', download=True)

#     files = glob.glob(os.path.join(disassembledPath,"*.png"))
    
#     # pool = ParallelPool(nodes=20)    
#     # results = pool.amap(real_esrgan, files, [model]*len(files), [improvedPath] * len(files))
#     results = t_map(real_esrgan, files, [model]*len(files), [improvedPath] * len(files))

# def real_esrgan(img_path, model, improvedPath):
#     image = Image.open(img_path).convert('RGB')
#     sr_image = model.predict(image)
#     img_name = os.path.basename(img_path)
#     sr_image.save(os.path.join(improvedPath, img_name))	

# Example usage:
# improve('input_folder_path', 'output_folder_path')
