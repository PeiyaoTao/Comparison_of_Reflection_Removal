# Traditional Method
# Carolina Li, Peiyao Tao
# CS 7180 Advanced Perception
# 10/15/2025
# This file implements the traditional ghosting reflection removal method and evaluates it on subsets of VOC2012 and Wildscene datasets.

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
import torchmetrics
from tqdm import tqdm
from numba import jit
import argparse


class ReflectionDataset(Dataset):
    """
    Dataset class for the synthetic VOC2012 data.
    """
    def __init__(self, root_dir, json_path):
        self.root_dir = root_dir
        self.blended_dir = os.path.join(root_dir, 'blended')
        self.transmission_dir = os.path.join(root_dir, 'transmission_layer')
        self.reflection_dir = os.path.join(root_dir, 'reflection_layer')
        try:
            with open(json_path, 'r') as f: self.image_pairs = json.load(f)
        except FileNotFoundError:
            self.image_pairs = []
            print(f"Warning: JSON file not found at {json_path}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        try:
            blended_img = Image.open(os.path.join(self.blended_dir, pair_info['blended'])).convert('RGB')
            transmission_img = Image.open(os.path.join(self.transmission_dir, pair_info['transmission_layer'])).convert('RGB')
            reflection_img = Image.open(os.path.join(self.reflection_dir, pair_info['reflection_layer'])).convert('RGB')
            
            return {
                'blended': TF.to_tensor(blended_img),
                'transmission': TF.to_tensor(transmission_img),
                'reflection': TF.to_tensor(reflection_img)
            }
        except (OSError, IOError):
            return None

class WildSceneDataset(Dataset):
    """
    Dataset class for the real-world Wildscene test set.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_pairs = []
        try:
            for scene_folder in sorted(os.listdir(root_dir), key=int):
                scene_path = os.path.join(root_dir, scene_folder)
                if os.path.isdir(scene_path):
                    mixed_path = os.path.join(scene_path, 'm.jpg')
                    gt_path = os.path.join(scene_path, 'g.jpg')
                    reflection_path = os.path.join(scene_path, 'r.jpg')
                    if os.path.exists(mixed_path) and os.path.exists(gt_path):
                        self.image_pairs.append({'blended': mixed_path, 'transmission': gt_path, 'reflection': reflection_path})
        except (FileNotFoundError, ValueError):
            self.image_pairs = []
            print(f"Warning: Wildscene directory not found or incorrectly structured at {root_dir}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        try:
            blended_img = Image.open(pair_info['blended']).convert('RGB')
            transmission_img = Image.open(pair_info['transmission']).convert('RGB')
            reflection_img = Image.open(pair_info['reflection']).convert('RGB')
            
            return {
                'blended': TF.to_tensor(blended_img),
                'transmission': TF.to_tensor(transmission_img),
                'reflection': TF.to_tensor(reflection_img)
            }
        except (OSError, IOError):
            return None

@jit(nopython=True)
def numba_convolve2d(image, kernel):
    """A simple 2D convolution implementation using Numba for speed."""
    h, w = image.shape; kh, kw = kernel.shape
    output = np.zeros_like(image); pad_h, pad_w = kh // 2, kw // 2
    for y in range(h):
        for x in range(w):
            sum_val = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = y - pad_h + ky, x - pad_w + kx
                    iy = max(0, min(iy, h - 1)); ix = max(0, min(ix, w - 1))
                    sum_val += image[iy, ix] * kernel[ky, kx]
            output[y, x] = sum_val
    return output

@jit(nopython=True)
def cost_function_ghosting_jit(TR_flat, image_ch, kernel, lambda_T, lambda_R, h, w, sobel_x, sobel_y):
    """ Cost function for ghosting reflection removal. """
    T = TR_flat[:h*w].reshape((h, w))
    R = TR_flat[h*w:].reshape((h, w))
    R_convolved = numba_convolve2d(R, kernel)
    error = image_ch - T - R_convolved
    fidelity = np.sum(error**2)
    T_grad_x, T_grad_y = numba_convolve2d(T, sobel_x), numba_convolve2d(T, sobel_y)
    R_grad_x, R_grad_y = numba_convolve2d(R, sobel_x), numba_convolve2d(R, sobel_y)
    T_smoothness = np.sum(T_grad_x**2) + np.sum(T_grad_y**2)
    R_smoothness = np.sum(R_grad_x**2) + np.sum(R_grad_y**2)
    return fidelity + lambda_T * T_smoothness + lambda_R * R_smoothness

@jit(nopython=True)
def gradient_ghosting_jit(TR_flat, image_ch, kernel, lambda_T, lambda_R, h, w, sobel_x, sobel_y):
    """ Gradient of the cost function for ghosting reflection removal. """
    T = TR_flat[:h*w].reshape((h, w)); R = TR_flat[h*w:].reshape((h, w))
    R_convolved = numba_convolve2d(R, kernel)
    error = image_ch - T - R_convolved
    grad_fidelity_T = -2 * error
    kernel_reversed = np.flip(kernel)
    grad_fidelity_R = -2 * numba_convolve2d(error, kernel_reversed)
    sobel_x_rev, sobel_y_rev = np.flip(sobel_x), np.flip(sobel_y)
    grad_T_smoothness = 2 * (numba_convolve2d(numba_convolve2d(T, sobel_x), sobel_x_rev) + numba_convolve2d(numba_convolve2d(T, sobel_y), sobel_y_rev))
    grad_R_smoothness = 2 * (numba_convolve2d(numba_convolve2d(R, sobel_x), sobel_x_rev) + numba_convolve2d(numba_convolve2d(R, sobel_y), sobel_y_rev))
    grad_T = grad_fidelity_T + lambda_T * grad_T_smoothness
    grad_R = grad_fidelity_R + lambda_R * grad_R_smoothness
    return np.concatenate((grad_T.flatten(), grad_R.flatten()))

class TraditionalGhostingMethod:
    """ 
    Traditional Ghosting reflection removal method implementation. 
    """
    def __init__(self, scale_size=300, max_iterations=30, num_passes=2, lambda_T=0.002, lambda_R=0.5, gamma=0.8):
        self.scale_size = scale_size
        self.max_iterations = max_iterations
        self.num_passes = num_passes
        self.lambda_T = lambda_T
        self.lambda_R = lambda_R
        self.gamma = gamma

    def __call__(self, image_tensor):
        """ Process the input image tensor and return estimated transmission and reflection layers. """
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_uint8 = (image_np * 255).astype(np.uint8)

        h, w, _ = image_uint8.shape
        scale = self.scale_size / max(h, w)
        new_dims = (int(w * scale), int(h * scale))
        image_resized_uint8 = cv2.resize(image_uint8, new_dims)
        image_float = image_resized_uint8.astype(np.float32) / 255.0

        kernel = self._estimate_kernel(image_resized_uint8)
        
        channels = cv2.split(image_float)
        T_channels = [0.5 * ch for ch in channels]
        R_channels = [0.5 * ch for ch in channels]

        for _ in range(self.num_passes):
            new_T_channels, new_R_channels = [], []
            for i in range(3):
                T_ch, R_ch = self._separate_layers_ghosting(channels[i], kernel, T_channels[i], R_channels[i])
                new_T_channels.append(T_ch); new_R_channels.append(R_ch)
            T_channels, R_channels = new_T_channels, new_R_channels

        T_final = cv2.merge(T_channels)
        R_final = cv2.merge(R_channels)
        
        T_final_clipped = np.clip(T_final, 0, 1).astype(np.float32)
        T_brightened = np.power(T_final_clipped, 1.0 / self.gamma).astype(np.float32)
        
        return torch.from_numpy(T_brightened).permute(2, 0, 1), \
               torch.from_numpy(R_final).permute(2, 0, 1)

    def _estimate_kernel(self, image):
        """ Estimate the ghosting kernel from the input image. """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        dy, dx = self._find_displacement(laplacian)
        c_k = self._find_attenuation(gray_image, (dy, dx))
        k_size = max(abs(dy), abs(dx)) * 2 + 1
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        center = k_size // 2
        kernel[center, center] = 1.0
        kernel[center + dy, center + dx] = c_k
        return kernel

    def _find_displacement(self, lap_image):
        """ Find displacement using autocorrelation of the Laplacian image. """
        f = np.fft.fft2(lap_image)
        autocorr = np.fft.ifft2(f * np.conj(f))
        autocorr = np.fft.fftshift(np.real(autocorr))
        h, w = autocorr.shape
        center_y, center_x = h // 2, w // 2
        mask = np.ones_like(autocorr, dtype=bool)
        mask[center_y-5:center_y+6, center_x-5:center_x+6] = False
        autocorr[~mask] = -np.inf # Ensure center peak is ignored
        peak_y, peak_x = np.unravel_index(np.argmax(autocorr), autocorr.shape)
        dy, dx = peak_y - center_y, peak_x - center_x
        max_disp = 25
        dy, dx = np.clip(dy, -max_disp, max_disp), np.clip(dx, -max_disp, max_disp)
        return dy, dx

    def _find_attenuation(self, gray_image, displacement):
        """ Find attenuation coefficient using corner variances. """
        dy, dx = displacement
        if dy == 0 and dx == 0: return 0.5
        gray_uint8 = (gray_image * 255).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(gray_uint8, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if corners is None: return 0.5
        ratios = []
        patch_size=5; half_patch=patch_size//2
        for i in corners:
            x, y = map(int, i.ravel())
            y1, x1 = (y-half_patch, y+half_patch+1), (x-half_patch, x+half_patch+1)
            y2, x2 = (y1[0]+dy, y1[1]+dy), (x1[0]+dx, x1[1]+dx)
            if all(val >= 0 for val in [y1[0], x1[0], y2[0], x2[0]]) and \
               y1[1] <= gray_image.shape[0] and x1[1] <= gray_image.shape[1] and \
               y2[1] <= gray_image.shape[0] and x2[1] <= gray_image.shape[1]:
                p1, p2 = gray_image[y1[0]:y1[1], x1[0]:x1[1]], gray_image[y2[0]:y2[1], x2[0]:x2[1]]
                var1, var2 = np.var(p1), np.var(p2)
                if var1 > 1e-5 and var2 > 1e-5:
                    ratios.append(np.sqrt(min(var1, var2)/max(var1, var2)))
        return np.mean(ratios) if ratios else 0.5

    def _separate_layers_ghosting(self, image_ch, kernel, T_init, R_init):
        """ Separate transmission and reflection layers using optimization. """
        h, w = image_ch.shape
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        TR_init_flat = np.concatenate((T_init.flatten(), R_init.flatten()))
        bounds = [(0, 1)] * len(TR_init_flat)
        
        result = minimize(cost_function_ghosting_jit, TR_init_flat, method='L-BFGS-B', 
                          jac=gradient_ghosting_jit, bounds=bounds,
                          args=(image_ch, kernel, self.lambda_T, self.lambda_R, h, w, sobel_x, sobel_y), 
                          options={'maxiter': self.max_iterations, 'disp': False})
        
        return result.x[:h*w].reshape((h, w)), result.x[h*w:].reshape((h, w))


def evaluate_traditional_method(traditional_model, dataloader, device, dataset_name):
    """ Evaluate the traditional ghosting method on a given dataset. """
    if len(dataloader.dataset) == 0:
        print(f"Skipping evaluation for {dataset_name} as no data was found.")
        return

    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
    
    print(f"\nEvaluating Traditional Method on {dataset_name}.")
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Testing on {dataset_name}"):
            if not data: continue
            blended, gt_transmission = data['blended'].to(device), data['transmission'].to(device)
            pred_t_trad, _ = traditional_model(blended.squeeze(0))
            h, w = gt_transmission.shape[-2:]
            pred_t_trad = TF.resize(pred_t_trad.unsqueeze(0), size=[h, w]).to(device)
            psnr_metric.update(pred_t_trad, gt_transmission)
            ssim_metric.update(pred_t_trad, gt_transmission)

    avg_psnr = psnr_metric.compute(); avg_ssim = ssim_metric.compute()
    print(f"\nResults for {dataset_name}:")
    print(f"Average PSNR: {avg_psnr:.4f} dB | Average SSIM: {avg_ssim:.4f}")
    
    print(f"\nDisplaying a few examples from {dataset_name}...")
    for i, data in enumerate(dataloader):
        if i >= 3: break
        if not data: continue
        blended, t_vis, r_vis = data['blended'], data['transmission'], data['reflection']
        p_t_vis, p_r_vis = traditional_model(blended.squeeze(0))

        if p_t_vis.dim() == 3: p_t_vis = p_t_vis.unsqueeze(0)
        if t_vis.dim() == 3: t_vis = t_vis.unsqueeze(0)

        h, w = t_vis.shape[-2:]
        p_t_vis_resized = TF.resize(p_t_vis, size=[h, w])

        p_t_vis_device = p_t_vis_resized.to(device)
        t_vis_device = t_vis.to(device)

        image_psnr = psnr_metric(p_t_vis_device, t_vis_device)
        image_ssim = ssim_metric(p_t_vis_device, t_vis_device)

        gt_images = {"Input Image": blended.squeeze(0), "GT Transmission": t_vis.squeeze(0), "GT Reflection": r_vis.squeeze(0)}
        fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
        fig1.suptitle("Ground Truth Comparison", fontsize=16)
        for ax, (title, img) in zip(axs1, gt_images.items()):
            ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1)); ax.set_title(title); ax.axis('off')
        plt.show()

        pred_images = {"Predicted Transmission": p_t_vis.squeeze(0), "Predicted Reflection": p_r_vis.squeeze(0)}
        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))
        fig2.suptitle(f"Traditional Method Predictions (PSNR: {image_psnr:.2f} dB, SSIM: {image_ssim:.4f})", fontsize=16)
        for ax, (title, img) in zip(axs2, pred_images.items()):
            ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1)); ax.set_title(title); ax.axis('off')
        plt.show()

def main(args):
    """ Main function to run evaluations on specified datasets. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    traditional_model = TraditionalGhostingMethod()
    print("Traditional Ghosting method model instantiated.")

    val_dataset = ReflectionDataset(root_dir=args.voc_path, json_path=args.voc_json)
    if len(val_dataset) > 0:
        num_samples = min(args.subset_size, len(val_dataset))
        indices = range(num_samples)
        val_subset = Subset(val_dataset, indices)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        evaluate_traditional_method(traditional_model, val_loader, device, f"VOC2012 Validation Set (Subset of {num_samples})")

    wildscene_test_dataset = WildSceneDataset(root_dir=args.wildscene_path)
    if len(wildscene_test_dataset) > 0:
        indices = range(num_samples)

        wildscene_subset = Subset(wildscene_test_dataset, indices)
        test_loader = DataLoader(wildscene_subset, batch_size=1, shuffle=False)
        evaluate_traditional_method(traditional_model, test_loader, device, f"Wildscene Test Set (Subset of {num_samples})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Traditional Ghosting Reflection Removal Method on a subset of data.")
    parser.add_argument('--voc_path', type=str, default="./VOC2012", help="Path to the root of the VOC2012 dataset.")
    parser.add_argument('--voc_json', type=str, default="val_list.json", help="Path to the validation JSON file for VOC2012.")
    parser.add_argument('--wildscene_path', type=str, default="./Wildscene/Wildscene", help="Path to the root of the Wildscene dataset.")
    parser.add_argument('--subset_size', type=int, default=20, help="Number of images to use from each dataset for evaluation.")
    
    args = parser.parse_args()
    main(args)