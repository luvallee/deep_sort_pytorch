import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tifffile
import numpy as np
import os


#masks_dir = "/home/vallee/Documents/deep_sort_pytorch/outputs/Fluo-C2DL-Huh7/Training_02"
masks_dir = "/home/vallee/Documents/deep_sort_pytorch/outputs/PhC-C2DH-U373/Training_01"

mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
masks = []
for f in mask_files:
    mask = cv2.imread(os.path.join(masks_dir, f))
    masks.append(mask)


# set the output video file name and codec
output_path = '/home/vallee/Documents/deep_sort_pytorch/outputs/PhC-C2DH-U373'

video_path = os.path.join(output_path, "track_PhC_tr_01.mp4")
fps = 5

video_frames = []

for i, mask in enumerate(masks):
    print(f"Traitement image {i}")

    if len(mask.shape) > 2:
        mask = np.squeeze(mask)

    video_frames.append(mask)

height, width, _ = video_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for frame in video_frames:
    out.write(frame)

out.release()
print(f"Vidéo enregistrée dans : {video_path}")