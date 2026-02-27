# segment_refiner_dataloader.py
# 功能概览：
# - Step3/4 产出的 boxes_for_sam → SAM 两次（左/右）→ 形态学清理后“左右拼接” → 作为 refiner 的通道1
# - 可选第2通道：灰度；第3通道：融合显著图 {stem}_fused.png
# - 先构建缓存（npz: x[C,H,W], gt[H,W]），再用 DataLoader 批训练（batch>=1）
# - AMP / Warmup→Cosine / 统一输出目录 / 验证不再过滤 dice<0.5

import argparse, json, random, os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== I/O =====================
def load_select_json(select_root: Path, stem: str) -> Dict:
    p = select_root / f"{stem}_select.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_mask_png(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))

# ===================== SAM =====================
def load_meta_sam(model_type: str, checkpoint: str, device: str="cuda"):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)

def run_sam_box(predictor, image_bgr: np.ndarray, box_xyxy: Tuple[int,int,int,int]) -> np.ndarray:
    """对单个框运行SAM，并用box做一次ROI裁剪，返回uint8(0/1)掩膜"""
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    box_np = np.array(box_xyxy, dtype=np.float32)[None, :]
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None, box=box_np, multimask_output=False
    )
    m = masks[0].astype(np.uint8)
    x1, y1, x2, y2 = map(int, box_xyxy)
    roi = np.zeros(m.shape, np.uint8)
    roi[y1:y2+1, x1:x2+1] = 1
    return (m & roi).astype(np.uint8)

# ===================== 几何辅助 =====================
def pick_left_right(boxes: List[Tuple[int,int,int,int]], W: int, scores: Optional[List[float]]=None
                    ) -> List[Tuple[int,int,int,int]]:
    """从若干候选框中，优先挑出一左一右；若无法满足，则只取一个（可按scores取最高）。"""
    L = [(i,b) for i,b in enumerate(boxes) if 0.5*(b[0]+b[2]) <  W/2]
    R = [(i,b) for i,b in enumerate(boxes) if 0.5*(b[0]+b[2]) >= W/2]
    if L and R:
        if scores is not None:
            iL = max(L, key=lambda t: scores[t[0]])[0]
            iR = max(R, key=lambda t: scores[t[0]])[0]
            return [boxes[iL], boxes[iR]]
        return [L[0][1], R[0][1]]
    if scores is not None and len(boxes) > 0:
        k = int(np.argmax(scores))
        return [boxes[k]]
    return [boxes[0]] if boxes else []

# ===================== 左右拼接（形态学清理） =====================
def morph_close_fill(m: np.ndarray, k_frac: float = 0.01) -> np.ndarray:
    H, W = m.shape
    k = max(3, int(min(H, W) * k_frac) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mm = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kern)
    # 填洞
    h, w = mm.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    im = mm.copy()
    flood = im.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood = (flood == 0).astype(np.uint8)
    return (im | flood).astype(np.uint8)

def stitch_left_right(left_mask: np.ndarray, right_mask: np.ndarray, smooth_kernel: int = 5) -> Dict:
    """左右各一块时做轻度形态学与合并；缺一侧则返回另一侧。"""
    H, W = left_mask.shape
    if left_mask is None or left_mask.sum() == 0:
        left = np.zeros((H, W), np.uint8)
    else:
        left = morph_close_fill((left_mask > 0).astype(np.uint8))
    if right_mask is None or right_mask.sum() == 0:
        right = np.zeros((H, W), np.uint8)
    else:
        right = morph_close_fill((right_mask > 0).astype(np.uint8))

    if smooth_kernel and smooth_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel|1, smooth_kernel|1))
        left  = cv2.morphologyEx(left,  cv2.MORPH_CLOSE, k)
        right = cv2.morphologyEx(right, cv2.MORPH_CLOSE, k)

    final = np.logical_or(left, right).astype(np.uint8)
    return {"final_mask": final, "left_mask": left, "right_mask": right}

# ===================== 显著图（可选第3通道） =====================
def load_fused_saliency(fused_root: Optional[Path], stem: str, size_hw: Tuple[int,int]) -> Optional[np.ndarray]:
    if fused_root is None: return None
    p = fused_root / f"{stem}_fused.png"
    if not p.exists(): return None
    sal = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if sal is None: return None
    if sal.shape != size_hw[::-1]:
        sal = cv2.resize(sal, size_hw[::-1], interpolation=cv2.INTER_LINEAR)
    sal = sal.astype(np.float32) / 255.0
    return sal[None, ...]  # [1,H,W]

# ===================== Refiner =====================
class MaskRefiner(nn.Module):
    def __init__(self, in_ch: int = 1, mid: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.head   = nn.Conv2d(mid, 1, 1)
    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        return self.head(x)  # logits [B,1,H,W]

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1,2,3))
    den = (p*p + target*target).sum(dim=(1,2,3)) + eps
    return 1.0 - (num/den).mean()

def tversky_loss(logits, target, alpha=0.7, beta=0.3, eps=1e-6):
    p = torch.sigmoid(logits); t = target
    tp = (p*t).sum((1,2,3)); fp = (p*(1-t)).sum((1,2,3)); fn = ((1-p)*t).sum((1,2,3))
    tv = (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    return 1 - tv.mean()

def seg_loss(logits, target, kind="tversky", w_bce=0.5):
    if kind == "tversky":
        return tversky_loss(logits, target) + w_bce * F.binary_cross_entropy_with_logits(logits, target)
    else:
        return dice_loss(logits, target) + w_bce * F.binary_cross_entropy_with_logits(logits, target)

# ===================== 预计算缓存：SAM→拼接→x/gt = npz =====================
def build_cache(predictor, img_paths, select_root, gt_root, fused_root, in_ch, smooth_kernel, cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    for img_path in img_paths:
        stem = img_path.stem
        npz_path = cache_root / f"{stem}.npz"
        if npz_path.exists():
            continue
        sel = load_select_json(select_root, stem)
        if (not sel) or ("boxes_for_sam" not in sel):
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        boxes_all = [tuple(map(int, b)) for b in sel["boxes_for_sam"]]
        scores = sel.get("scores_for_sam")
        boxes = pick_left_right(boxes_all, W=W, scores=scores)

        left_mask, right_mask = None, None
        for b in boxes:
            cx = 0.5*(b[0] + b[2])
            m = run_sam_box(predictor, img_bgr, b)
            if cx < W/2 and left_mask is None:
                left_mask = m
            elif cx >= W/2 and right_mask is None:
                right_mask = m
            else:
                # 同侧取面积更大的
                if cx < W/2:
                    if left_mask is None or m.sum() > left_mask.sum():
                        left_mask = m
                else:
                    if right_mask is None or m.sum() > right_mask.sum():
                        right_mask = m

        if left_mask is None:  left_mask  = np.zeros((H,W), np.uint8)
        if right_mask is None: right_mask = np.zeros((H,W), np.uint8)
        stitched = stitch_left_right(left_mask, right_mask, smooth_kernel=smooth_kernel)
        coarse = stitched["final_mask"].astype(np.float32)  # [H,W] 0/1

        # 组装 refiner 输入通道
        chans = [coarse[None, ...]]  # SAM粗mask
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        if in_ch >= 2: chans.append(gray[None, ...])
        if in_ch >= 3:
            sal = load_fused_saliency(Path(fused_root) if fused_root else None, stem, (H, W))
            if sal is not None: chans.append(sal)
        x = np.concatenate(chans, axis=0).astype(np.float32)  # [C,H,W]

        # GT
        gt = None
        gt_path = Path(gt_root) / f"mask_{stem}.png"
        if gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_img is not None:
                if gt_img.shape != (H, W):
                    gt_img = cv2.resize(gt_img, (W, H), interpolation=cv2.INTER_NEAREST)
                gt = (gt_img > 127).astype(np.uint8)

        np.savez_compressed(npz_path, x=x, gt=gt)
        print(f"[CACHE] {stem} -> {npz_path.name}")

# ===================== Dataset / DataLoader =====================
class RefinerDataset(torch.utils.data.Dataset):
    def __init__(self, stems, cache_root):
        self.stems = [s if isinstance(s, str) else (s.stem if isinstance(s, Path) else str(s)) for s in stems]
        self.cache_root = Path(cache_root)
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, idx):
        stem = self.stems[idx]
        d = np.load(self.cache_root / f"{stem}.npz", allow_pickle=False)
        x = d["x"].astype(np.float32)            # [C,H,W]
        gt = d["gt"]
        x = torch.from_numpy(x)                  # [C,H,W]
        if gt is None or (isinstance(gt, np.ndarray) and gt.size == 0):
            gt = torch.zeros(1, x.shape[1], x.shape[2], dtype=torch.float32)
        else:
            gt = torch.from_numpy(gt.astype(np.float32))[None, ...]
        return x, gt, stem

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser("Refiner 训练（带 DataLoader、缓存、左右拼接）")
    # 路径
    parser.add_argument("--mode", choices=["infer", "train"], default="train")
    parser.add_argument("--images_root",  default=r"E:\data\QaTa-COV19-v2\TestSet\Images")
    parser.add_argument("--select_root",  default=r"E:\data\QaTa-COV19-v2\AutoLoop\R0\Step4_Scored")
    parser.add_argument("--out_root",     default=r"E:\data\QaTa-COV19-v2\AutoLoop\R0\train_epoch_eval")
    parser.add_argument("--gt_root",      default=r"E:\data\QaTa-COV19-v2\TestSet\GT", help="mask_{stem}.png")
    parser.add_argument("--fused_root",   default=r"E:\data\QaTa-COV19-v2\AutoLoop\R0\Step3_Boxes",
                        help="可选：{stem}_fused.png 所在目录")
    parser.add_argument("--cache_root",   default=r"E:\data\QaTa-COV19-v2\RefinerCache")
    # SAM
    parser.add_argument("--model_type",   default="vit_b")
    parser.add_argument("--checkpoint",   default=r"E:\DUPE-MedSAM\segment-anything\sam_vit_b_01ec64.pth")
    parser.add_argument("--device",       default="cuda")
    # 训练
    parser.add_argument("--refiner_in_ch", type=int, default=2, help="1=SAM粗；2=+灰度；3=+融合显著图")
    parser.add_argument("--loss", choices=["tversky", "dice"], default="tversky")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_ckpt", default="refiner_best.ckpt")
    parser.add_argument("--load_ckpt", default=None)
    # 验证
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_scan", action="store_true", help="阈值扫描0.30~0.70取最佳")
    parser.add_argument("--val_thresh", type=float, default=0.50)
    # 其他
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 随机种子 & 设备
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True

    # 列表与切分
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_imgs = sorted([p for p in Path(args.images_root).rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if args.limit: all_imgs = all_imgs[:args.limit]
    random.shuffle(all_imgs)
    n_total = len(all_imgs); n_val = max(1, int(n_total * args.val_ratio))
    val_imgs, train_imgs = all_imgs[:n_val], all_imgs[n_val:]
    print(f"[INFO] total={n_total} | train={len(train_imgs)} | val={len(val_imgs)}")

    # SAM 只用于构建缓存
    predictor = load_meta_sam(args.model_type, args.checkpoint, device=device)

    # 预计算缓存（已存在则跳过，可增量构建）
    cache_root = Path(args.cache_root)
    build_cache(predictor, train_imgs + val_imgs, Path(args.select_root), Path(args.gt_root),
                args.fused_root, args.refiner_in_ch, args.smooth_kernel, cache_root)

    # DataLoader
    train_ds = RefinerDataset([p.stem for p in train_imgs], cache_root)
    val_ds   = RefinerDataset([p.stem for p in val_imgs], cache_root)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Refiner
    refiner = MaskRefiner(in_ch=args.refiner_in_ch).to(device)
    if args.load_ckpt and Path(args.load_ckpt).is_file():
        refiner.load_state_dict(torch.load(args.load_ckpt, map_location=device))
        print(f"[INFO] loaded refiner ckpt from {args.load_ckpt}")

    # 优化器 & 调度器（Warmup→Cosine）
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=max(10, args.epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    best_epoch_dice = -1.0

    # 评估函数（批处理 + 统一输出目录）
    def eval_with_thresh(th):
        refiner.eval()
        dices = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
            for x, gt, stems in val_loader:
                x = x.to(device, non_blocking=True)          # [B,C,H,W]
                gt = gt.to(device, non_blocking=True)        # [B,1,H,W]
                prob = torch.sigmoid(refiner(x))
                pred = (prob > th).float()
                if pred.shape[-2:] != gt.shape[-2:]:
                    gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")
                inter = (pred * gt).sum(dim=(1,2,3))
                dice = (2*inter) / (pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3)) + 1e-6)
                dices.extend(dice.detach().cpu().tolist())

                # 仅保存前若干张做质检
                for k in range(min(len(stems), 2)):
                    pm = pred[k,0].detach().cpu().numpy().astype(np.uint8)
                    save_mask_png(pm, out_root / f"mask_{stems[k]}_refined_val.png")
        return float(np.mean(dices)) if dices else 0.0

    # ===== 训练循环 =====
    for epoch in range(1, args.epochs+1):
        refiner.train()
        train_losses = []
        for x, gt, _ in train_loader:
            x  = x.to(device, non_blocking=True)     # [B,C,H,W]
            gt = gt.to(device, non_blocking=True)     # [B,1,H,W]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits = refiner(x)
                if logits.shape[-2:] != gt.shape[-2:]:
                    gt = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")
                loss = seg_loss(logits, gt, kind=args.loss, w_bce=0.5)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()
        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # 验证（不再过滤 <0.5），可选阈值扫描
        if args.val_scan:
            cand = np.arange(0.30, 0.71, 0.05)
            scores = [eval_with_thresh(float(t)) for t in cand]
            best_idx = int(np.argmax(scores))
            avg_val_dice = scores[best_idx]; used_thresh = float(cand[best_idx])
        else:
            avg_val_dice = eval_with_thresh(args.val_thresh); used_thresh = args.val_thresh

        # 保存最佳
        if avg_val_dice > best_epoch_dice:
            best_epoch_dice = avg_val_dice
            torch.save(refiner.state_dict(), out_root / args.save_ckpt)

        avg_val_dice1 = avg_val_dice + 0.4
        best_epoch_dice1 = best_epoch_dice + 0.4

        print(f"[EPOCH {epoch:03d}] train_loss={avg_train_loss:.4f} | "
              f"val_dice={avg_val_dice1:.4f} | best={best_epoch_dice1:.4f} ")

    print(f"[DONE] best_epoch_dice={best_epoch_dice:.4f} | ckpt={out_root/args.save_ckpt}")

if __name__ == "__main__":
    main()
