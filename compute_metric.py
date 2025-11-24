"""
Image Quality Metrics Computation Script

This script computes multiple image quality metrics (PSNR, SSIM, LPIPS, FID) 
for comparing ground truth images with predicted/reconstructed images.

Supports:
- Single-run evaluation: pred_folder/*.png
- Multi-run evaluation: pred_folder/run0/*.png, pred_folder/run1/*.png, etc.

For multi-run cases, the best sample for each image is selected based on 
the specified metric (PSNR, SSIM, or LPIPS).

Usage:
    python compute_metric.py --gt_folder <path> --pred_folder <path> [options]

Example:
    python compute_metric.py --gt_folder dataset/test-ffhq --pred_folder results/samples --plot
"""

from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

from data import ImageDataset
from evaluate_fid import calculate_fid

def load_images_from_folder(folder_path, resolution=256):
    """
    Load images from a folder and convert to tensors in range [-1, 1]
    
    Args:
        folder_path: path to folder containing images
        resolution: target resolution for images
    
    Returns:
        torch.Tensor: [N, C, H, W] in range [-1, 1]
    """
    folder = Path(folder_path)
    image_files = sorted(folder.glob('*.png')) + sorted(folder.glob('*.jpg'))
    
    images = []
    for img_path in tqdm(image_files, desc=f"Loading images from {folder.name}"):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((resolution, resolution), Image.BILINEAR)
        img = transforms.ToTensor()(img)
        img = img * 2 - 1
        images.append(img)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {folder_path}")
    
    return torch.stack(images)


def organize_multi_run_images(dir_pred_images, num_gt, resolution=256):
    """
    Organize images from multiple runs into a single tensor
    
    Args:
        dir_pred_images: Directory containing run0, run1, etc. subfolders
        num_gt: Number of ground truth images (for validation)
        resolution: Target resolution for images
    
    Expected structure:
        dir_pred_images/run0/image_0.png ... image_N.png
        dir_pred_images/run1/image_0.png ... image_N.png
    
    Returns:
        torch.Tensor: [num_runs, num_gt, C, H, W] in range [-1, 1]
    """
    dir_pred = Path(dir_pred_images)
    
    # Support both 'run_0' and 'run0' naming conventions
    run_folders = sorted([f for f in dir_pred.iterdir() 
                         if f.is_dir() and (f.name.startswith('run_') or 
                                           (f.name.startswith('run') and len(f.name) > 3 and f.name[3:].replace('_','').isdigit()))])
    
    if len(run_folders) == 0:
        raise ValueError(f"No run_* or run* folders found in {dir_pred_images}")
    
    num_runs = len(run_folders)
    print(f"Found {num_runs} run folders: {[f.name for f in run_folders]}")
    
    all_runs = []
    for run_folder in run_folders:
        images = load_images_from_folder(run_folder, resolution)
        
        if len(images) != num_gt:
            raise ValueError(
                f"Folder {run_folder.name} has {len(images)} images, "
                f"but expected {num_gt} images"
            )
        
        all_runs.append(images)
    
    organized = torch.stack(all_runs, dim=0)
    
    print(f"Organized multi-run images: {organized.shape}")
    return organized

def select_best_samples(gt_images, pred_images_multi_run, metric='psnr', device='cuda'):
    """
    Select the best sample from multiple runs based on the specified metric
    
    Args:
        gt_images: Ground truth images, shape [N, C, H, W]
        pred_images_multi_run: Predicted images from multiple runs, shape [num_runs, N, C, H, W]
        metric: Metric to use for selection - 'psnr' (higher better), 'ssim' (higher better), or 'lpips' (lower better)
        device: Device for computation, 'cuda' or 'cpu'
    
    Returns:
        best_samples: Selected best samples, shape [N, C, H, W]
        best_indices: Indices of the best run for each image, shape [N]
    """
    num_runs, num_images = pred_images_multi_run.shape[:2]
    best_samples = []
    best_indices = []
    
    print(f"\nSelecting best samples based on {metric.upper()}...")
    
    if metric.lower() == 'psnr':
        for img_idx in tqdm(range(num_images), desc="Computing PSNR for selection"):
            psnr_scores = []
            for runidx in range(num_runs):
                pred = pred_images_multi_run[runidx, img_idx:img_idx+1]
                gt = gt_images[img_idx:img_idx+1]
                psnr = compute_psnr(gt, pred)[0]
                psnr_scores.append(psnr)
            
            best_run = np.argmax(psnr_scores)
            best_samples.append(pred_images_multi_run[best_run, img_idx])
            best_indices.append(best_run)
    
    elif metric.lower() == 'lpips':
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        
        for img_idx in tqdm(range(num_images), desc="Computing LPIPS for selection"):
            lpips_scores = []
            for runidx in range(num_runs):
                pred = pred_images_multi_run[runidx, img_idx:img_idx+1].to(device)
                gt = gt_images[img_idx:img_idx+1].to(device)
                with torch.no_grad():
                    lpips_score = lpips_fn(pred, gt).item()
                lpips_scores.append(lpips_score)
            
            best_run = np.argmin(lpips_scores)
            best_samples.append(pred_images_multi_run[best_run, img_idx])
            best_indices.append(best_run)

    elif metric.lower() == 'ssim':
        for img_idx in tqdm(range(num_images), desc="Computing SSIM for selection"):
            ssim_scores = []
            for runidx in range(num_runs):
                pred = pred_images_multi_run[runidx, img_idx:img_idx+1]
                gt = gt_images[img_idx:img_idx+1]
                ssim = compute_ssim(gt, pred)[0]
                ssim_scores.append(ssim)
            
            best_run = np.argmax(ssim_scores)
            best_samples.append(pred_images_multi_run[best_run, img_idx])
            best_indices.append(best_run)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    best_samples = torch.stack(best_samples)
    best_indices = np.array(best_indices)
    
    print(f"Best run distribution: {np.bincount(best_indices, minlength=num_runs)}")
    
    return best_samples, best_indices
    

def compute_psnr(gt_images, pred_images):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for image pairs
    
    Args:
        gt_images: Ground truth images, shape [N, C, H, W] in range [-1, 1]
        pred_images: Predicted images, shape [N, C, H, W] in range [-1, 1]
    
    Returns:
        np.array: PSNR values for each image pair in dB
    """
    gt = (gt_images + 1) / 2
    pred = (pred_images + 1) / 2
    
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    psnr_values = []
    for i in range(len(gt_np)):
        gt_img = np.transpose(gt_np[i], (1, 2, 0))
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
        psnr_values.append(psnr)
    
    return np.array(psnr_values)

def compute_ssim(gt_images, pred_images):
    """
    Compute Structural Similarity Index (SSIM) for image pairs
    
    Args:
        gt_images: Ground truth images, shape [N, C, H, W] in range [-1, 1]
        pred_images: Predicted images, shape [N, C, H, W] in range [-1, 1]
    
    Returns:
        np.array: SSIM values for each image pair in range [0, 1]
    """
    gt = (gt_images + 1) / 2
    pred = (pred_images + 1) / 2
    
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    ssim_values = []
    for i in range(len(gt_np)):
        gt_img = np.transpose(gt_np[i], (1, 2, 0))
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        ssim = structural_similarity(
            gt_img, pred_img, 
            data_range=1.0, 
            multichannel=True,
            channel_axis=2,
            win_size=11
        )
        ssim_values.append(ssim)
    
    return np.array(ssim_values)

def compute_lpips(gt_images, pred_images, device='cuda'):
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS) for image pairs
    
    Args:
        gt_images: Ground truth images, shape [N, C, H, W] in range [-1, 1]
        pred_images: Predicted images, shape [N, C, H, W] in range [-1, 1]
        device: Device for computation, 'cuda' or 'cpu'
    
    Returns:
        np.array: LPIPS distances for each image pair (lower is better)
    """
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    gt = gt_images.to(device)
    pred = pred_images.to(device)
    
    lpips_values = []
    batch_size = 10

    for i in range(0, len(gt), batch_size):
        batch_gt = gt[i:i+batch_size]
        batch_pred = pred[i:i+batch_size]
        
        with torch.no_grad():
            lpips_batch = lpips_fn(batch_gt, batch_pred)
        
        lpips_values.extend(lpips_batch.cpu().numpy().flatten())
    
    return np.array(lpips_values)

def compute_fid(gt_folder, pred_folder, device='cuda', batch_size=50):
    """
    Compute Fréchet Inception Distance (FID) between two image distributions
    
    Args:
        gt_folder: Path to folder containing ground truth images
        pred_folder: Path to folder containing predicted images
        device: Device for computation, 'cuda' or 'cpu'
        batch_size: Batch size for processing
    
    Returns:
        float: FID score (lower is better), or None if computation fails
    """
    try:
        gt_dataset = ImageDataset(root=gt_folder, resolution=256)
        pred_dataset = ImageDataset(root=pred_folder, resolution=256)
        
        gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        fid_score = calculate_fid(gt_loader, pred_loader)
        
        return fid_score
    except Exception as e:
        print(f"Warning: FID calculation failed with error: {e}")
        print("Skipping FID calculation...")
        return None
    
def print_statistics(metric_name, values):
    """
    Print statistical summary of metric values
    
    Args:
        metric_name: Name of the metric
        values: Array of metric values
    """
    print(f"\n{metric_name}:")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Std:  {np.std(values):.4f}")
    print(f"  Min:  {np.min(values):.4f}")
    print(f"  Max:  {np.max(values):.4f}")
    print(f"  Median: {np.median(values):.4f}")

def save_best_samples_for_fid(pred_images):
    """
    Save best prediction samples to a temporary folder for FID computation.
    This is needed when using multi-run structure since FID requires a folder of images.
    
    Args:
        pred_images: Array of shape [N, C, H, W] containing selected best images
        
    Returns:
        Path to the temporary folder containing saved images
    """
    import tempfile
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="fid_calc_"))
    
    # Save each image
    for i, img in enumerate(pred_images):
        # Convert PyTorch tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert from [C, H, W] to [H, W, C]
        img_hwc = img.transpose(1, 2, 0)
        
        # Denormalize from [-1, 1] to [0, 255]
        img_hwc = ((img_hwc + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        
        # Save image with padded filename
        filename = f"{i:05d}.png"
        filepath = temp_dir / filename
        
        # Use PIL to save
        from PIL import Image
        img_pil = Image.fromarray(img_hwc)
        img_pil.save(filepath)
    
    return temp_dir

def save_results(output_path, psnr_values, ssim_values, lpips_values, fid_score, 
                best_indices=None, num_runs=1):
    """
    Save comprehensive metric results to a text file
    
    Args:
        output_path: Path to save the results file
        psnr_values: Array of PSNR values
        ssim_values: Array of SSIM values
        lpips_values: Array of LPIPS values
        fid_score: FID score (can be None)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Image Quality Metrics\n")
        f.write("=" * 60 + "\n\n")
        
        # Multi-run info
        if num_runs > 1:
            f.write(f"Multi-run evaluation: {num_runs} runs\n")
            f.write("Selection method: Best sample per image based on PSNR\n\n")
        
        # PSNR
        f.write("PSNR (Peak Signal-to-Noise Ratio):\n")
        f.write(f"  Mean ± Std: {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f}\n")
        f.write(f"  Range: [{np.min(psnr_values):.4f}, {np.max(psnr_values):.4f}]\n")
        f.write(f"  Median: {np.median(psnr_values):.4f}\n\n")
        
        # SSIM
        f.write("SSIM (Structural Similarity Index):\n")
        f.write(f"  Mean ± Std: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}\n")
        f.write(f"  Range: [{np.min(ssim_values):.4f}, {np.max(ssim_values):.4f}]\n")
        f.write(f"  Median: {np.median(ssim_values):.4f}\n\n")
        
        # LPIPS
        f.write("LPIPS (Learned Perceptual Image Patch Similarity):\n")
        f.write(f"  Mean ± Std: {np.mean(lpips_values):.4f} ± {np.std(lpips_values):.4f}\n")
        f.write(f"  Range: [{np.min(lpips_values):.4f}, {np.max(lpips_values):.4f}]\n")
        f.write(f"  Median: {np.median(lpips_values):.4f}\n\n")
        
        # FID
        if fid_score is not None:
            f.write("FID (Fréchet Inception Distance):\n")
            f.write(f"  Score: {fid_score:.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Per-image values:\n")
        f.write("=" * 60 + "\n")
        
        # Add best run index column if multi-run
        if best_indices is not None:
            f.write(f"{'Index':<8} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12} {'Best Run':<10}\n")
            f.write("-" * 70 + "\n")
            for i in range(len(psnr_values)):
                f.write(f"{i:<8} {psnr_values[i]:<12.4f} {ssim_values[i]:<12.4f} {lpips_values[i]:<12.4f} {best_indices[i]:<10}\n")
            
            # Add run selection statistics
            f.write("\n" + "=" * 60 + "\n")
            f.write("Run Selection Statistics:\n")
            f.write("=" * 60 + "\n")
            unique_runs, counts = np.unique(best_indices, return_counts=True)
            for runidx, count in zip(unique_runs, counts):
                percentage = (count / len(best_indices)) * 100
                f.write(f"  Run {runidx}: {count} images ({percentage:.1f}%)\n")
        else:
            f.write(f"{'Index':<8} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12}\n")
            f.write("-" * 60 + "\n")
            for i in range(len(psnr_values)):
                f.write(f"{i:<8} {psnr_values[i]:<12.4f} {ssim_values[i]:<12.4f} {lpips_values[i]:<12.4f}\n")

def plot_metrics(psnr_values, ssim_values, lpips_values, output_path, best_indices=None):
    """
    Generate and save visualization of metric distributions and trends
    
    Creates a 2x2 grid with:
    - PSNR histogram
    - SSIM histogram
    - LPIPS histogram
    - Line plot showing all metrics across images (or run selection if multi-run)
    
    Args:
        psnr_values: Array of PSNR values
        ssim_values: Array of SSIM values
        lpips_values: Array of LPIPS values
        output_path: Path to save the figure
        best_indices: Array of best run indices (for multi-run evaluation)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR histogram
    axes[0, 0].hist(psnr_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(psnr_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(psnr_values):.2f}')
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('PSNR Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[0, 1].hist(ssim_values, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(ssim_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(ssim_values):.3f}')
    axes[0, 1].set_xlabel('SSIM')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('SSIM Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # LPIPS histogram
    axes[1, 0].hist(lpips_values, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(lpips_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(lpips_values):.3f}')
    axes[1, 0].set_xlabel('LPIPS')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('LPIPS Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fourth plot: line plot or run selection
    if best_indices is not None:
        # Multi-run: show run selection distribution
        unique_runs, counts = np.unique(best_indices, return_counts=True)
        axes[1, 1].bar(unique_runs, counts, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Run Index')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Best Run Selection Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for runidx, count in zip(unique_runs, counts):
            percentage = (count / len(best_indices)) * 100
            axes[1, 1].text(runidx, count, f'{percentage:.1f}%', 
                          ha='center', va='bottom', fontsize=9)
    else:
        # Single run: line plot of all metrics over images
        axes[1, 1].plot(psnr_values / np.max(psnr_values), 'b-o', label='PSNR (normalized)', markersize=4)
        axes[1, 1].plot(ssim_values, 'g-s', label='SSIM', markersize=4)
        axes[1, 1].plot(lpips_values, 'r-^', label='LPIPS', markersize=4)
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('Metrics Across Images')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nMetrics plot saved to: {output_path}")
    plt.close()



def main():
    """
    Main function to compute image quality metrics from command line
    
    Parses arguments, loads images, computes metrics, and saves results.
    Automatically detects single-run vs multi-run folder structure.
    """
    parser = argparse.ArgumentParser(description='Compute image quality metrics')
    parser.add_argument('--gt_folder', type=str, required=True, 
                        help='Path to ground truth images folder')
    parser.add_argument('--pred_folder', type=str, required=True, 
                        help='Path to predicted/reconstructed images folder')
    parser.add_argument('--output', type=str, default='metrics_results.txt',
                        help='Output file path for results')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Image resolution (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')
    parser.add_argument('--skip_fid', action='store_true',
                        help='Skip FID calculation')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Computing Image Quality Metrics")
    print("=" * 60)
    print(f"Ground Truth Folder: {args.gt_folder}")
    print(f"Prediction Folder: {args.pred_folder}")
    print(f"Resolution: {args.resolution}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Load ground truth images
    print("\nLoading ground truth images...")
    gt_images = load_images_from_folder(args.gt_folder, args.resolution)
    num_gt = len(gt_images)
    print(f"Loaded {num_gt} ground truth images")
    
    # Check for multi-run structure
    pred_folder = Path(args.pred_folder)
    # Support both 'run_0' and 'run0' naming conventions
    run_folders = sorted([f for f in pred_folder.iterdir() 
                         if f.is_dir() and (f.name.startswith('run_') or 
                                           (f.name.startswith('run') and len(f.name) > 3 and f.name[3:].replace('_','').isdigit()))])
    
    best_indices = None
    num_runs = 1
    
    if len(run_folders) > 0:
        # Multi-run case
        print(f"\nDetected multi-run structure with {len(run_folders)} runs")
        
        # Organize multi-run images
        pred_images_multi_run = organize_multi_run_images(args.pred_folder, num_gt, args.resolution)
        num_runs = pred_images_multi_run.shape[0]
        
        # Select best samples based on PSNR
        pred_images, best_indices = select_best_samples(
            gt_images, pred_images_multi_run, 
            metric='psnr',
            device=args.device
        )
    else:
        # Single run case
        print("\nNo run* folders detected, treating as single run")
        print("Loading prediction images...")
        
        try:
            pred_images = load_images_from_folder(args.pred_folder, args.resolution)
        except ValueError as e:
            print(f"\nError: {e}")
            print(f"\nPlease check the folder structure:")
            print(f"  Option 1 (multi-run): {args.pred_folder}/run0/, {args.pred_folder}/run1/, ...")
            print(f"  Option 2 (single run): {args.pred_folder}/*.png")
            return
        
        if len(pred_images) != num_gt:
            raise ValueError(
                f"Number of prediction images ({len(pred_images)}) does not match "
                f"ground truth images ({num_gt})"
            )
    
    print(f"\nUsing {len(pred_images)} images for metric computation")
    
    print("\nComputing PSNR...")
    psnr_values = compute_psnr(gt_images, pred_images)
    print_statistics("PSNR", psnr_values)
    
    print("\nComputing SSIM...")
    ssim_values = compute_ssim(gt_images, pred_images)
    print_statistics("SSIM", ssim_values)
    
    print("\nComputing LPIPS...")
    lpips_values = compute_lpips(gt_images, pred_images, device=args.device)
    print_statistics("LPIPS", lpips_values)
    
    fid_score = None
    if not args.skip_fid:
        print("\nComputing FID...")
        try:
            if num_runs > 1:
                # Multi-run: save best samples to temp folder for FID calculation
                temp_dir = save_best_samples_for_fid(pred_images)
                fid_score = compute_fid(args.gt_folder, str(temp_dir), device=args.device)
                
                # Clean up temp folder
                import shutil
                shutil.rmtree(temp_dir)
            else:
                # Single run: use pred_folder directly
                fid_score = compute_fid(args.gt_folder, args.pred_folder, 
                                       device=args.device)
            
            if fid_score is not None:
                print(f"FID Score: {fid_score:.4f}")
        except Exception as e:
            print(f"FID calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nSaving results...")
    save_results(args.output, psnr_values, ssim_values, lpips_values, fid_score,
                best_indices=best_indices, num_runs=num_runs)
    print(f"Results saved to: {args.output}")
    
    if args.plot:
        plot_path = Path(args.output).with_suffix('.png')
        plot_metrics(psnr_values, ssim_values, lpips_values, plot_path, 
                    best_indices=best_indices)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    if num_runs > 1:
        print(f"Number of runs: {num_runs}")
        print(f"Selection metric: PSNR")
    print(f"PSNR:  {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f}")
    print(f"SSIM:  {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"LPIPS: {np.mean(lpips_values):.4f} ± {np.std(lpips_values):.4f}")
    if fid_score is not None:
        print(f"FID:   {fid_score:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()