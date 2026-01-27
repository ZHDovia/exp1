import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn
import bm3d
import os
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))

output_folder = os.path.join(current_dir, "results")
os.makedirs(output_folder, exist_ok=True)

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.05):
    noisy = img.copy()
    rnd = np.random.random(img.shape)
    noisy[rnd < prob/2] = 0
    noisy[rnd > 1-prob/2] = 255
    return noisy.astype(np.uint8)

def real_bm3d_denoise(img, sigma=25):
    img_float = img.astype(np.float32) / 255.0
    denoised = bm3d.bm3d(img_float, sigma_psd=sigma/255.0)
    return (denoised * 255).clip(0, 255).astype(np.uint8)

def calculate_psnr(original, denoised):
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, denoised, window_size=11, k1=0.01, k2=0.03):
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2
    mu1 = cv2.GaussianBlur(original.astype(np.float64), (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(denoised.astype(np.float64), (window_size, window_size), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(original.astype(np.float64) ** 2, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(denoised.astype(np.float64) ** 2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original.astype(np.float64) * denoised.astype(np.float64), (window_size, window_size), 1.5) - mu1_mu2
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return np.mean(ssim_map)

img_path = os.path.join(current_dir, "1.jpg")
original = cv2.imread(img_path, 0)

if original is None:
    try:
        from PIL import Image
        pil_img = Image.open(img_path).convert('L')
        original = np.array(pil_img)
    except:
        print(f"无法读取图片: {img_path}")
        exit()

gaussian_noisy = add_gaussian_noise(original, sigma=75)
saltpepper_noisy = add_salt_pepper_noise(original, prob=0.10)

gaussian_result = real_bm3d_denoise(gaussian_noisy, 75)
saltpepper_result = real_bm3d_denoise(saltpepper_noisy, 60)

cv2.imwrite(os.path.join(output_folder, "gaussian_noisy.jpg"), gaussian_noisy)
cv2.imwrite(os.path.join(output_folder, "saltpepper_noisy.jpg"), saltpepper_noisy)
cv2.imwrite(os.path.join(output_folder, "gaussian_result_bm3d.jpg"), gaussian_result)
cv2.imwrite(os.path.join(output_folder, "saltpepper_result_bm3d.jpg"), saltpepper_result)

psnr_gaussian_noisy = calculate_psnr(original, gaussian_noisy)
ssim_gaussian_noisy = calculate_ssim(original, gaussian_noisy)
psnr_gaussian_result = calculate_psnr(original, gaussian_result)
ssim_gaussian_result = calculate_ssim(original, gaussian_result)

psnr_saltpepper_noisy = calculate_psnr(original, saltpepper_noisy)
ssim_saltpepper_noisy = calculate_ssim(original, saltpepper_noisy)
psnr_saltpepper_result = calculate_psnr(original, saltpepper_result)
ssim_saltpepper_result = calculate_ssim(original, saltpepper_result)

print("高斯噪声图像质量评价:")
print(f"噪声图像PSNR: {psnr_gaussian_noisy:.2f} dB")
print(f"噪声图像SSIM: {ssim_gaussian_noisy:.4f}")
print(f"去噪图像PSNR: {psnr_gaussian_result:.2f} dB")
print(f"去噪图像SSIM: {ssim_gaussian_result:.4f}")
print(f"PSNR提升: {psnr_gaussian_result - psnr_gaussian_noisy:.2f} dB")
print(f"SSIM提升: {ssim_gaussian_result - ssim_gaussian_noisy:.4f}")

print("\n椒盐噪声图像质量评价:")
print(f"噪声图像PSNR: {psnr_saltpepper_noisy:.2f} dB")
print(f"噪声图像SSIM: {ssim_saltpepper_noisy:.4f}")
print(f"去噪图像PSNR: {psnr_saltpepper_result:.2f} dB")
print(f"去噪图像SSIM: {ssim_saltpepper_result:.4f}")
print(f"PSNR提升: {psnr_saltpepper_result - psnr_saltpepper_noisy:.2f} dB")
print(f"SSIM提升: {ssim_saltpepper_result - ssim_saltpepper_noisy:.4f}")
