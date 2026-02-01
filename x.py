import cv2
import numpy as np
import bm3d
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))


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

def add_mixed_noise(img, gauss_sigma=25, sp_prob=0.05):
    gauss_noisy = add_gaussian_noise(img, gauss_sigma)
    return add_salt_pepper_noise(gauss_noisy, sp_prob)


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def ista_denoise_with_reference(noisy_img, clean_img, lambda_, max_iter=100, step_size=1.0):
    y = noisy_img.astype(np.float32) / 255.0
    x = y.copy()
    psnr_history = []
    
    for i in range(max_iter):
        x_old = x.copy()
        gradient = x - y
        x = x - step_size * gradient
        x = soft_threshold(x, lambda_ * step_size)
        
        current_denoised = (x * 255).clip(0, 255).astype(np.uint8)
        psnr = calculate_psnr(clean_img, current_denoised)
        psnr_history.append(psnr)
        
        if i > 10 and abs(psnr_history[-1] - psnr_history[-2]) < 0.01:
            break
    
    denoised = (x * 255).clip(0, 255).astype(np.uint8)
    return denoised, psnr_history


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
    return 20 * np.log10(max_pixel / np.sqrt(mse))

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
    return np.mean(numerator / denominator)


def run_core_experiment(original_img):
    results = {
        'noise_type': [],
        'gauss_level': [],
        'sp_level': [],
        'algorithm': [],
        'psnr': [],
        'ssim': [],
        'time': [],
        'denoised_img': []
    }
    
    core_configs = [
        {'type': 'gaussian', 'gauss': 50, 'sp': 0, 'desc': '纯高斯噪声 σ=50'},
        {'type': 'mixed', 'gauss': 25, 'sp': 0.1, 'desc': '轻度混合 σ=25 prob=0.1'},
        {'type': 'mixed', 'gauss': 50, 'sp': 0.2, 'desc': '中度混合 σ=50 prob=0.2'},
        {'type': 'mixed', 'gauss': 75, 'sp': 0.3, 'desc': '重度混合 σ=75 prob=0.3'},
    ]
    
    print("ISTA与BM3D混合噪声去噪对比实验开始")
    print("")
    
    for config in core_configs:
        noise_type = config['type']
        gauss_level = config['gauss']
        sp_level = config['sp']
        
        print(f"测试配置: {config['desc']}")
        
        if noise_type == 'mixed':
            noisy_img = add_mixed_noise(original_img, gauss_sigma=gauss_level, sp_prob=sp_level)
        else:
            noisy_img = add_gaussian_noise(original_img, sigma=gauss_level)
        
        
        noisy_psnr = calculate_psnr(original_img, noisy_img)
        noisy_ssim = calculate_ssim(original_img, noisy_img)
        print(f"噪声图像质量: PSNR={noisy_psnr:.2f}dB, SSIM={noisy_ssim:.4f}")
        
        
        if noise_type == 'gaussian':
            lambda_ = 0.15
            max_iter = 100
        else:
            lambda_ = 0.15 * (gauss_level / 25) + 0.08 * (sp_level / 0.1)
            lambda_ = max(0.1, min(lambda_, 0.3))
            max_iter = 120
        
        start_time = time.time()
        ista_result, psnr_history = ista_denoise_with_reference(
            noisy_img, original_img, 
            lambda_=lambda_, 
            max_iter=max_iter,
            step_size=1.0
        )
        ista_time = time.time() - start_time
        
        ista_psnr = calculate_psnr(original_img, ista_result)
        ista_ssim = calculate_ssim(original_img, ista_result)
        
        
        results['noise_type'].append(noise_type)
        results['gauss_level'].append(gauss_level)
        results['sp_level'].append(sp_level)
        results['algorithm'].append('ISTA')
        results['psnr'].append(ista_psnr)
        results['ssim'].append(ista_ssim)
        results['time'].append(ista_time)
        results['denoised_img'].append(ista_result)
        
        print(f"ISTA结果: PSNR={ista_psnr:.2f}dB, SSIM={ista_ssim:.4f}, 时间={ista_time:.3f}s")
        print(f"  PSNR提升: {ista_psnr - noisy_psnr:+.2f}dB")
        
        
        start_time = time.time()
        bm3d_sigma = gauss_level if noise_type == 'gaussian' else max(gauss_level, 60)
        bm3d_result = real_bm3d_denoise(noisy_img, sigma=bm3d_sigma)
        bm3d_time = time.time() - start_time
        
        bm3d_psnr = calculate_psnr(original_img, bm3d_result)
        bm3d_ssim = calculate_ssim(original_img, bm3d_result)
        
        
        results['noise_type'].append(noise_type)
        results['gauss_level'].append(gauss_level)
        results['sp_level'].append(sp_level)
        results['algorithm'].append('BM3D')
        results['psnr'].append(bm3d_psnr)
        results['ssim'].append(bm3d_ssim)
        results['time'].append(bm3d_time)
        results['denoised_img'].append(bm3d_result)
        
        print(f"BM3D结果: PSNR={bm3d_psnr:.2f}dB, SSIM={bm3d_ssim:.4f}, 时间={bm3d_time:.3f}s")
        print(f"  PSNR提升: {bm3d_psnr - noisy_psnr:+.2f}dB")
        print(f"  BM3D优于ISTA: {bm3d_psnr - ista_psnr:+.2f}dB")
        print(f"  时间比率(BM3D/ISTA): {bm3d_time/ista_time:.1f}倍")
        print("")
        
        
        if noise_type == 'mixed' and gauss_level == 50 and sp_level == 0.2:
            convergence_history = psnr_history
    
    return results, convergence_history

import cv2
import numpy as np
import bm3d
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))


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

def add_mixed_noise(img, gauss_sigma=25, sp_prob=0.05):
    gauss_noisy = add_gaussian_noise(img, gauss_sigma)
    return add_salt_pepper_noise(gauss_noisy, sp_prob)


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def ista_denoise_with_reference(noisy_img, clean_img, lambda_, max_iter=100, step_size=1.0):
    y = noisy_img.astype(np.float32) / 255.0
    x = y.copy()
    
    for i in range(max_iter):
        x_old = x.copy()
        gradient = x - y
        x = x - step_size * gradient
        x = soft_threshold(x, lambda_ * step_size)
        
        if np.linalg.norm(x - x_old) / np.linalg.norm(x_old) < 1e-5:
            break
    
    denoised = (x * 255).clip(0, 255).astype(np.uint8)
    return denoised

def ista_denoise_with_history(noisy_img, clean_img, lambda_, max_iter=100, step_size=1.0):
    y = noisy_img.astype(np.float32) / 255.0
    x = y.copy()
    psnr_history = []
    
    for i in range(max_iter):
        x_old = x.copy()
        gradient = x - y
        x = x - step_size * gradient
        x = soft_threshold(x, lambda_ * step_size)
        
        current_denoised = (x * 255).clip(0, 255).astype(np.uint8)
        psnr = calculate_psnr(clean_img, current_denoised)
        psnr_history.append(psnr)
        
        if i > 10 and abs(psnr_history[-1] - psnr_history[-2]) < 0.01:
            break
    
    denoised = (x * 255).clip(0, 255).astype(np.uint8)
    return denoised, psnr_history


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
    return 20 * np.log10(max_pixel / np.sqrt(mse))

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
    return np.mean(numerator / denominator)


def run_core_experiment(original_img):
    results = {
        'noise_type': [],
        'gauss_level': [],
        'sp_level': [],
        'algorithm': [],
        'psnr': [],
        'ssim': [],
        'time': [],
        'denoised_img': []
    }
    
    core_configs = [
        {'type': 'gaussian', 'gauss': 50, 'sp': 0},
        {'type': 'mixed', 'gauss': 25, 'sp': 0.1},
        {'type': 'mixed', 'gauss': 50, 'sp': 0.2},
        {'type': 'mixed', 'gauss': 75, 'sp': 0.3},
    ]
    
    for config in core_configs:
        noise_type = config['type']
        gauss_level = config['gauss']
        sp_level = config['sp']
        
        if noise_type == 'mixed':
            noisy_img = add_mixed_noise(original_img, gauss_sigma=gauss_level, sp_prob=sp_level)
        else:
            noisy_img = add_gaussian_noise(original_img, sigma=gauss_level)
        
        
        if noise_type == 'gaussian':
            lambda_ = 0.15
            max_iter = 100
        else:
            lambda_ = 0.15 * (gauss_level / 25) + 0.08 * (sp_level / 0.1)
            lambda_ = max(0.1, min(lambda_, 0.3))
            max_iter = 120
        
        start_time = time.time()
        
        if noise_type == 'mixed' and gauss_level == 50 and sp_level == 0.2:
            ista_result, psnr_history = ista_denoise_with_history(
                noisy_img, original_img, 
                lambda_=lambda_, 
                max_iter=max_iter,
                step_size=1.0
            )
        else:
            ista_result = ista_denoise_with_reference(
                noisy_img, original_img, 
                lambda_=lambda_, 
                max_iter=max_iter,
                step_size=1.0
            )
            psnr_history = []
        
        ista_time = time.time() - start_time
        
        ista_psnr = calculate_psnr(original_img, ista_result)
        ista_ssim = calculate_ssim(original_img, ista_result)
        
        
        results['noise_type'].append(noise_type)
        results['gauss_level'].append(gauss_level)
        results['sp_level'].append(sp_level)
        results['algorithm'].append('ISTA')
        results['psnr'].append(ista_psnr)
        results['ssim'].append(ista_ssim)
        results['time'].append(ista_time)
        results['denoised_img'].append(ista_result)
        
        
        start_time = time.time()
        bm3d_sigma = gauss_level if noise_type == 'gaussian' else max(gauss_level, 60)
        bm3d_result = real_bm3d_denoise(noisy_img, sigma=bm3d_sigma)
        bm3d_time = time.time() - start_time
        
        bm3d_psnr = calculate_psnr(original_img, bm3d_result)
        bm3d_ssim = calculate_ssim(original_img, bm3d_result)
        
        
        results['noise_type'].append(noise_type)
        results['gauss_level'].append(gauss_level)
        results['sp_level'].append(sp_level)
        results['algorithm'].append('BM3D')
        results['psnr'].append(bm3d_psnr)
        results['ssim'].append(bm3d_ssim)
        results['time'].append(bm3d_time)
        results['denoised_img'].append(bm3d_result)
    
    return results, psnr_history if 'psnr_history' in locals() else []
def run_core_experiment(original_img):
    results = {
        'noise_type': [],
        'gauss_level': [],
        'sp_level': [],
        'algorithm': [],
        'psnr': [],
        'ssim': [],
        'time': [],
        'denoised_img': []
    }
    
    core_configs = [
        {'type': 'gaussian', 'gauss': 50, 'sp': 0},
        {'type': 'mixed', 'gauss': 25, 'sp': 0.1},
        {'type': 'mixed', 'gauss': 50, 'sp': 0.2},
        {'type': 'mixed', 'gauss': 75, 'sp': 0.3},
    ]
    
    for config in core_configs:
        noise_type = config['type']
        gauss_level = config['gauss']
        sp_level = config['sp']
        
        if noise_type == 'mixed':
            noisy_img = add_mixed_noise(original_img, gauss_sigma=gauss_level, sp_prob=sp_level)
        else:
            noisy_img = add_gaussian_noise(original_img, sigma=gauss_level)
        
        
        output_folder = os.path.join(current_dir, "results")
        os.makedirs(output_folder, exist_ok=True)
        
        
        if noise_type == 'gaussian':
            lambda_ = 0.15
            max_iter = 100
        else:
            lambda_ = 0.15 * (gauss_level / 25) + 0.08 * (sp_level / 0.1)
            lambda_ = max(0.1, min(lambda_, 0.3))
            max_iter = 120
        
        start_time = time.time()
        
        if noise_type == 'mixed' and gauss_level == 50 and sp_level == 0.2:
            ista_result, psnr_history = ista_denoise_with_history(
                noisy_img, original_img, 
                lambda_=lambda_, 
                max_iter=max_iter,
                step_size=1.0
            )
        else:
            ista_result = ista_denoise_with_reference(
                noisy_img, original_img, 
                lambda_=lambda_, 
                max_iter=max_iter,
                step_size=1.0
            )
            psnr_history = []
        
        ista_time = time.time() - start_time
        
        ista_psnr = calculate_psnr(original_img, ista_result)
        ista_ssim = calculate_ssim(original_img, ista_result)
        
        results['noise_type'].append(noise_type)
        results['gauss_level'].append(gauss_level)
        results['sp_level'].append(sp_level)
        results['algorithm'].append('ISTA')
        results['psnr'].append(ista_psnr)
        results['ssim'].append(ista_ssim)
        results['time'].append(ista_time)
        results['denoised_img'].append(ista_result)
        
        start_time = time.time()
        bm3d_sigma = gauss_level if noise_type == 'gaussian' else max(gauss_level, 60)
        bm3d_result = real_bm3d_denoise(noisy_img, sigma=bm3d_sigma)
        bm3d_time = time.time() - start_time
        
        bm3d_psnr = calculate_psnr(original_img, bm3d_result)
        bm3d_ssim = calculate_ssim(original_img, bm3d_result)
        
        results['noise_type'].append(noise_type)
        results['gauss_level'].append(gauss_level)
        results['sp_level'].append(sp_level)
        results['algorithm'].append('BM3D')
        results['psnr'].append(bm3d_psnr)
        results['ssim'].append(bm3d_ssim)
        results['time'].append(bm3d_time)
        results['denoised_img'].append(bm3d_result)
        
        
        if noise_type == 'mixed' and gauss_level == 50 and sp_level == 0.2:
            cv2.imwrite(os.path.join(output_folder, "original.png"), original_img)
            cv2.imwrite(os.path.join(output_folder, "mixed_noisy.png"), noisy_img)
            cv2.imwrite(os.path.join(output_folder, "ista_result.png"), ista_result)
            cv2.imwrite(os.path.join(output_folder, "bm3d_result.png"), bm3d_result)
        
    
    return results, psnr_history if 'psnr_history' in locals() else []


def main():
    
    img_path = os.path.join(current_dir, "2.jpg")
    
    if not os.path.exists(img_path):
        return
    
    original = cv2.imread(img_path, 0)
    if original is None:
        try:
            from PIL import Image
            original = np.array(Image.open(img_path).convert('L'))
        except:
            original = np.ones((512, 512), dtype=np.uint8) * 128
    

    results, convergence_history = run_core_experiment(original)
    

    noise_types = np.array(results['noise_type'])
    gauss_levels = np.array(results['gauss_level'])
    sp_levels = np.array(results['sp_level'])
    algorithms = np.array(results['algorithm'])
    psnr_values = np.array(results['psnr'])
    ssim_values = np.array(results['ssim'])
    time_values = np.array(results['time'])
    

    for i in range(0, len(results['noise_type']), 2):
        noise_type = results['noise_type'][i]
        gauss_level = results['gauss_level'][i]
        sp_level = results['sp_level'][i]
        
        ista_psnr = results['psnr'][i]
        ista_ssim = results['ssim'][i]
        ista_time = results['time'][i]
        
        bm3d_psnr = results['psnr'][i+1]
        bm3d_ssim = results['ssim'][i+1]
        bm3d_time = results['time'][i+1]
        
        print(f"{noise_type}, σ={gauss_level}, prob={sp_level}")
        print(f"ISTA: PSNR={ista_psnr:.2f}dB, SSIM={ista_ssim:.4f}, Time={ista_time:.3f}s")
        print(f"BM3D: PSNR={bm3d_psnr:.2f}dB, SSIM={bm3d_ssim:.4f}, Time={bm3d_time:.3f}s")
        print(f"Difference: PSNR={bm3d_psnr-ista_psnr:+.2f}dB, TimeRatio={bm3d_time/ista_time:.1f}x")
        print("")

if __name__ == "__main__":
    main()
