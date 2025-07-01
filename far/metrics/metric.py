import lpips
import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image
import numpy as np
import io

from .fvd import FrechetVideoDistance


# larger: more incompressible
def jpeg_incompressibility(samples):
    print("Compressibility computing...")
    # samples: [b f c h w]
    # samples_ = samples[:, context_length:]

    if isinstance(samples, torch.Tensor):
        # Scale [0,1] floats to [0,255]
        # samples_ = (samples * 255.0).round().clamp(0, 255)
        # samples_ = samples_.to(torch.uint8).cpu().numpy()
        samples_ = samples.float().detach().cpu().numpy()
        if samples_.max() < 2:
            samples_ = (samples_.clip(0, 1) * 255).astype(np.uint8)
        else:
            samples_ = samples_.astype(np.uint8)

    # [b, f, c, h, w] -> [b, f, h, w, c]
    samples_ = samples_.transpose(0, 1, 3, 4, 2)

    batch_sizes = torch.zeros((samples.shape[0], 1)) # [b, 1]
    # Compute JPEG size per video
    for i, video in enumerate(samples_):
        total_bytes = 0
        for frame in video:
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            total_bytes += buffer.tell()
        # Convert bytes to kb
        batch_sizes[i] = total_bytes / 1024

    return batch_sizes

# larger: sizes smaller --> more compressible
def jpeg_compressibility(samples):
    sizes = jpeg_incompressibility(samples)
    return -sizes / 500


# Only select the most different patches in the frames for MSE comparison
# The larger top-k MSEs are the worse model is
def topk_patch_mse(
    samples: torch.Tensor,
    gt:      torch.Tensor,
    patch_div: int = 4,
    topk:      int = 5,
) -> torch.Tensor:
    """
    samples, gt: [B, F, C, H, W]
      – H and W must be divisible by patch_div.
    Returns:
      Tensor of shape [B, 1], one inconsistency score per batch element,
      where for each frame we:
        1) split into (patch_div x patch_div) non-overlapping patches of size (H/patch_div, W/patch_div),
        2) compute per-patch MSE between sample & gt,
        3) take the mean of the top-k worst patches,
      then average over frames.
    """
    B, fr, C, H, W = samples.shape
    assert H % patch_div == 0 and W % patch_div == 0, \
        f"H ({H}) and W ({W}) must be divisible by patch_div ({patch_div})"
    
    # patch height & width
    ph = H // patch_div
    pw = W // patch_div

    # merge batch & frame dims → [B*F, C, H, W]
    samples_ = samples.contiguous().view(B * fr, C, H, W)
    gt_      = gt.contiguous().view(B * fr, C, H, W)
    
    # extract non-overlapping (ph × pw) patches → [B*F, C*ph*pw, num_patches]
    xs = F.unfold(samples_, kernel_size=(ph, pw), stride=(ph, pw))
    ys = F.unfold(gt_,      kernel_size=(ph, pw), stride=(ph, pw))
    
    # number of patches per frame = patch_div^2
    N = xs.size(-1)
    
    # reshape to [B*F, N, C, ph, pw]
    xs = xs.view(B * fr, C, ph, pw, N).permute(0, 4, 1, 2, 3)
    ys = ys.view(B * fr, C, ph, pw, N).permute(0, 4, 1, 2, 3)
    
    # flatten per patch → [B*F, N, C*ph*pw], compute MSE over last dim → [B*F, N]
    diffs    = (xs - ys).reshape(B * fr, N, -1)
    patch_mse = diffs.pow(2).mean(dim=2)
    
    # for each of the B*F frames, take the mean of the top-k worst patches → [B*F]
    topk_vals     = torch.topk(patch_mse, k=topk, dim=1).values
    per_frame_mse = topk_vals.mean(dim=1)
    
    # reshape back to [B, F], average over frames → [B]
    per_batch_mse = per_frame_mse.view(B, fr).mean(dim=1)
    
    # return [B, 1]
    return per_batch_mse.unsqueeze(1)


class VideoMetric:

    def __init__(self, metric=['fvd', 'lpips', 'mse', 'psnr', 'ssim'], device='cuda'):
        self.metric_dict = {}
        self.device = device

        if 'mse' in metric:
            self.metric_dict['mse'] = True

        if 'psnr' in metric:
            self.metric_dict['psnr'] = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3])

        if 'ssim' in metric:
            self.metric_dict['ssim'] = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(self.device)

        if 'lpips' in metric:
            self.metric_dict['lpips'] = lpips.LPIPS(net='alex', spatial=False).to(self.device)  # [0, 1]

        if 'fvd' in metric:
            self.metric_dict['fvd'] = FrechetVideoDistance().to(self.device)

    @torch.no_grad()
    def compute(self, sample, gt, context_length):
        batch_size, num_trajectory, num_frame, channel, height, width = sample.shape
        # average from per video, then select max from num_trajectory, then average from batch
        result_dict = {}

        if 'mse' in self.metric_dict:
            mse = torch.mean((sample[:, :, context_length:] - gt[:, :, context_length:])**2, dim=(3, 4, 5))
            result_dict['mse'] = float(mse.mean(-1).max(-1)[0].mean())

        if 'psnr' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous()
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous()
            psnr = self.metric_dict['psnr'](sample_, gt_)
            psnr = rearrange(psnr, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['psnr'] = float(psnr.mean(-1).max(-1)[0].mean())

        if 'ssim' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous().to(self.device)
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous().to(self.device)
            ssim = torch.zeros(sample_.shape[0])
            for start in tqdm(range(0, sample_.shape[0], 256), desc='computing ssim'):
                ssim[start:start + 256] = self.metric_dict['ssim'](sample_[start:start + 256].to(self.device), gt_[start:start + 256].to(self.device))
            ssim = rearrange(ssim, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['ssim'] = float(ssim.mean(-1).max(-1)[0].mean())

        if 'lpips' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').float().contiguous()
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').float().contiguous()

            lpips = torch.zeros(sample_.shape[0], 1, 1, 1).to(self.device)
            for start in tqdm(range(0, sample_.shape[0], 256), desc='computing lpips'):
                lpips[start:start + 256] = self.metric_dict['lpips'](sample_[start:start + 256].to(self.device), gt_[start:start + 256].to(self.device))
            lpips = torch.mean(lpips, dim=(1, 2, 3))
            lpips = rearrange(lpips, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['lpips'] = float(lpips.mean(-1).min(-1)[0].mean())

        if 'fvd' in self.metric_dict and num_frame >= 10:
            sample_ = rearrange(sample, 'b n f c h w -> f b n c h w').float().contiguous()
            gt_ = rearrange(gt, 'b n f c h w -> f b n c h w').float().contiguous()
            sample_ = 2.0 * sample_ - 1
            gt_ = 2.0 * gt_ - 1

            fvd = torch.zeros(num_trajectory).to(self.device)
            for traj_idx in tqdm(range(num_trajectory), desc='computing fvd'):
                fvd[traj_idx] = self.metric_dict['fvd'].compute(sample_[:, :, traj_idx, ...], gt_[:, :, traj_idx, ...], device=self.device)
            result_dict['fvd'] = float(fvd.mean())

        return result_dict

    @torch.no_grad()
    def compute_reward(self, sample, gt, context_length):
        batch_size, num_trajectory, num_frame, channel, height, width = sample.shape
        # average from per video, then select max from num_trajectory, then average from batch
        result_dict = {}

        if 'mse' in self.metric_dict:
            mse = torch.mean((sample[:, :, context_length:] - gt[:, :, context_length:])**2, dim=(3, 4, 5))
            result_dict['mse'] = float(mse.mean(-1).max(-1)[0].mean())

        if 'psnr' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous()
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous()
            psnr = self.metric_dict['psnr'](sample_, gt_)
            psnr = rearrange(psnr, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['psnr'] = float(psnr.mean(-1).max(-1)[0].mean())

        if 'ssim' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous().to(self.device)
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous().to(self.device)
            ssim = torch.zeros(sample_.shape[0])
            for start in range(0, sample_.shape[0], 256):
                ssim[start:start + 256] = self.metric_dict['ssim'](sample_[start:start + 256].to(self.device), gt_[start:start + 256].to(self.device))
            ssim = rearrange(ssim, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['ssim'] = ssim.mean(-1).max(-1)[0].unsqueeze(1).float()

        if 'lpips' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').float().contiguous()
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').float().contiguous()

            lpips = torch.zeros(sample_.shape[0], 1, 1, 1).to(self.device)
            for start in range(0, sample_.shape[0], 256):
                lpips[start:start + 256] = self.metric_dict['lpips'](sample_[start:start + 256].to(self.device), gt_[start:start + 256].to(self.device))
            lpips = torch.mean(lpips, dim=(1, 2, 3))
            lpips = rearrange(lpips, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['lpips'] = lpips.mean(-1).min(-1)[0].unsqueeze(1).float()

        if 'fvd' in self.metric_dict and num_frame >= 10:
            sample_ = rearrange(sample, 'b n f c h w -> f b n c h w').float().contiguous()
            gt_ = rearrange(gt, 'b n f c h w -> f b n c h w').float().contiguous()
            sample_ = 2.0 * sample_ - 1
            gt_ = 2.0 * gt_ - 1

            fvd = torch.zeros(num_trajectory).to(self.device)
            for traj_idx in range(num_trajectory):
                fvd[traj_idx] = self.metric_dict['fvd'].compute(sample_[:, :, traj_idx, ...], gt_[:, :, traj_idx, ...], device=self.device)
            result_dict['fvd'] = float(fvd.mean())

        return result_dict
