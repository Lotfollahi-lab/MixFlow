import io
import os
import subprocess

import cv2
import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch
import umap
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid


def get_ram():
    mem = psutil.virtual_memory()
    free = mem.available / 1024 ** 3
    total = mem.total / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return ' [' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def get_vram():
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return ' [' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def format_time(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

def start_tensorboard(logdir):
    if os.name == 'nt':  # Windows
        process = subprocess.Popen(['tensorboard', '--logdir', logdir])
    else:  # Unix-based systems
        process = subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', '6006', '--host', '*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return process

def log_system_metrics(writer, step, mode='batch'):
    # CPU Usage
    cpu_usage = psutil.cpu_percent(interval=None)
    writer.add_scalar(f'System/{mode}/CPU Usage', cpu_usage, step)

    # RAM Usage
    ram_usage = psutil.virtual_memory().percent
    writer.add_scalar(f'System/{mode}/RAM Usage', ram_usage, step)

    # Disk Usage (ROM)
    #rom_usage = psutil.disk_usage('/').percent
    #writer.add_scalar(f'System/{mode}/ROM Usage', rom_usage, step)

    # GPU Usage (if available)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming you have one GPU
        writer.add_scalar(f'System/{mode}/GPU Usage', gpu.load * 100, step)
        writer.add_scalar(f'System/{mode}/GPU Memory Usage', gpu.memoryUsed / gpu.memoryTotal * 100, step)

def create_color_square(color):
    img = np.zeros((28, 28, 3), dtype=np.uint8)
    img[:, :] = color 
    return img

def create_angle_image(angle):
    img = np.zeros((28, 28, 3), dtype=np.uint8)
    center = (14, 14)
    start_point = (5, 14)
    end_point = (23, 14)
    img = cv2.line(img, start_point, end_point, (255, 255, 255), 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (28, 28))
    return rotated_img

def log_colors_and_angles(writer, colors, angles, step):
    # Log color squares
    color_images = [create_color_square(color) for color in colors]
    color_images = torch.tensor(np.stack(color_images)).permute(0, 3, 1, 2)  
    writer.add_images('Colors', color_images, step)

    # Log angle visualizations
    angle_images = [create_angle_image(angle) for angle in angles]
    angle_images = torch.tensor(np.stack(angle_images)).permute(0, 3, 1, 2)  
    writer.add_images('Angles', angle_images, step)

def log_batch_images(writer, batch_images, step):
    if batch_images.max() > 1:  
        batch_images = batch_images.float() / 255.0 
    grid = make_grid(batch_images, nrow=8)  
    writer.add_image('Batch Images', grid, step)

def log_umap_embeddings(writer, means, variances, step):
    means_flat = means.view(means.size(1), -1).cpu().numpy()
    variances_flat = variances.view(variances.size(1), -1).cpu().numpy()

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2)
    mean_embeddings = reducer.fit_transform(means_flat)
    variance_embeddings = reducer.fit_transform(variances_flat)

    labels = np.arange(means_flat.shape[0])

    writer.add_embedding(mean_embeddings, metadata=labels, tag='UMAP_Means', global_step=step)
    writer.add_embedding(variance_embeddings, metadata=labels, tag='UMAP_Variances', global_step=step)

def log_heatmap(writer, colors, angles, class_list, gumbel_softmax_output, step):
    heatmap_matrix = torch.sum(gumbel_softmax_output.cpu().detach(), dim=(2, 3, 4)).numpy()
    min_val = heatmap_matrix.min()
    max_val = heatmap_matrix.max()
    heatmap_matrix = (heatmap_matrix - min_val) / (max_val - min_val)

    row_labels = list()
    for i in range(len(class_list)):
        row_labels.append(f"D-{int(i%10)} C-{colors[i]}, A-{angles[i]}")

    fig, ax = plt.subplots(figsize=(15, 8))
    plot = sns.heatmap(heatmap_matrix, annot=False, fmt=".2f", cmap="coolwarm", ax=ax, yticklabels=row_labels)
    plt.subplots_adjust(left=0.3) #plt.tight_layout()
    ax.set_title("Class vs Modes Heatmap")
    ax.tick_params(axis='y', labelsize=10)

    # Convert plot to image and log it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    heatmap_image = Image.open(buf)
    heatmap_image_tensor = ToTensor()(heatmap_image)
    writer.add_image('Heatmap/Class_vs_Modes', heatmap_image_tensor, step)
    plt.close(fig)

