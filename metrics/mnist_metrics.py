import random
import warnings

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from PIL import Image, ImageOps
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm


def retrieve_mnist_images(label, num_images=10):
    # Returns the first `num_images` images from the MNIST dataset with the specified label
    transform = transforms.ToTensor()
    mnist_data = MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_images = [img for img, lbl in mnist_data if lbl == label]
    return mnist_images[:num_images]

def apply_transformations(image, angle, color):
    # Apply rotation and color transformations to the input image
    image = transforms.ToPILImage()(image)
    rotated_image = image.rotate(angle)
    rotated_image_rgb = ImageOps.colorize(rotated_image.convert("L"), black="black", white=(int(color[0]), int(color[1]), int(color[2])))
    return transforms.ToTensor()(rotated_image_rgb)

def compute_metrics(real_images, generated_images, metric_fn):
    metrics = []
    real_images = torch.stack(real_images).view(-1, 3, 28, 28).to(generated_images.device)    
    generated_images = generated_images.view(-1, 3, 28*28)
    real_images = real_images.view(-1, 3, 28*28)
    metric = metric_fn(generated_images, real_images)
    metrics.append(torch.mean(metric).item()) # this will give the mean distance between the RGB channels
    
    return sum(metrics) / len(metrics)  # Return the mean metric

def compute_FID(real_images, generated_images):
    real_images = torch.stack(real_images).view(-1, 3, 28, 28).to(generated_images.device)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_images.to(dtype=torch.uint8), real=True)
    fid.update(generated_images.to(dtype=torch.uint8), real=False)

    fid_score = fid.compute()
    return fid_score.item()

def compute_lpips(real_images, generated_images, loss_fn_alex):
    real_images = torch.stack(real_images).view(-1, 3, 28, 28).to(generated_images.device)
    real_images = F.interpolate(real_images, size=(256,256), mode='bilinear', align_corners=False)
    generated_images = F.interpolate(generated_images, size=(256,256), mode='bilinear', align_corners=False)  
    distances = []
    for real_img in real_images:
        for generated_img in generated_images:
            img0 = (real_img - 0.5) * 2
            img1 = (generated_img - 0.5) * 2
            distances.append(loss_fn_alex(img0, img1))       
    score = torch.mean(torch.stack(distances))
    return score.item()

def evaluate_generated_images(config, logger, generated_images, class_labels):
    logger.info("Evaluating generated images...")
    
    # Initialize the metric functions (W1, W2, MMD)
    w1_loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.01)
    w2_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    mmd_loss_fn = SamplesLoss(loss="gaussian", blur=0.5)
    
    metrics_results = []
    generated_images_chunks = torch.chunk(generated_images, len(class_labels))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        loss_fn_alex = lpips.LPIPS(net='alex').to(generated_images.device) 

    for i, (generated_images_chunk, class_label) in tqdm(enumerate(zip(generated_images_chunks, class_labels)), total=len(class_labels)):
        label = class_label[0]
        color = class_label[1:4]
        angle = class_label[4] * 360
        
        # Retrieve n real MNIST images of the same label
        real_images = retrieve_mnist_images(label, num_images=20)
        
        # Apply rotation and color transformations to real images
        transformed_real_images = [apply_transformations(img, angle, color) for img in real_images]
        
        # Compute the metrics
        w1_metric = compute_metrics(transformed_real_images, generated_images_chunk, w1_loss_fn)
        w2_metric = compute_metrics(transformed_real_images, generated_images_chunk, w2_loss_fn)
        mmd_metric = compute_metrics(transformed_real_images, generated_images_chunk, mmd_loss_fn)
        fid_score = compute_FID(transformed_real_images, generated_images_chunk)
        lpips_score = compute_lpips(transformed_real_images, generated_images_chunk, loss_fn_alex)
        
        metrics_results.append({
            "Label": label.item(),
            "R": int(color[0] * 255),
            "G": int(color[1] * 255),
            "B": int(color[2] * 255),
            "Angle": angle.item(),
            "W1": w1_metric,
            "W2": w2_metric,
            "MMD": mmd_metric,
            "FID": fid_score,
            "LPIPS": lpips_score
        })
    
    # Save metrics to a CSV file
    df = pd.DataFrame(metrics_results)
    df.to_csv(config['evaluation_output_path'], index=False)
    logger.info("Evaluation metrics saved successfully to: %s", config['evaluation_output_path'])

