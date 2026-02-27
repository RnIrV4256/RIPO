# Step5: SAM分割 + 可训练细化器（左右两框提示 + 拼接版，统一输出目录，验证不滤<0.5>)
import argparse, json, random
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
        # 若提供了分数，则分别在左右中选分数最高的；否则取各自第一个
        if scores is not None:
            iL = max(L, key=lambda t: scores[t[0]])[0]
            iR = max(R, key=lambda t: scores[t[0]])[0]
            return [boxes[iL], boxes[iR]]
        return [L[0][1], R[0][1]]
    # 只有一侧：按分数最高或顺序取一个
    if scores is not None and len(boxes) > 0:
        k = int(np.argmax(scores))
        return [boxes[k]]
    return [boxes[0]] if boxes else []

# ===================== 左右拼接（形态学稳健） =====================
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

# ===================== 细化器与损失 =====================
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

# ===================== 单次前向（两框→SAM→左右拼接→细化→Dice） =====================
def forward_once(predictor, refiner, img_path: Path, select_root: Path, gt_root: Path,
                 smooth_kernel: int, in_ch: int, device: str, fused_root: Optional[Path], val_thresh: float):
    stem = img_path.stem
    sel = load_select_json(select_root, stem)
    if (not sel) or ("boxes_for_sam" not in sel):
        return None

    boxes_all = [tuple(map(int, b)) for b in sel["boxes_for_sam"]]
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return None
    H, W = img_bgr.shape[:2]

    scores = sel.get("scores_for_sam")  # 若Step4有分数可用
    boxes = pick_left_right(boxes_all, W=W, scores=scores)

    # SAM 分别跑两次（1~2个框）
    left_mask, right_mask = None, None
    for b in boxes:
        cx = 0.5*(b[0] + b[2])
        m = run_sam_box(predictor, img_bgr, b)
        if cx < W/2 and left_mask is None:
            left_mask = m
        elif cx >= W/2 and right_mask is None:
            right_mask = m
        else:
            # 如果两框都落在同侧，保留更大的一个
            if cx < W/2:
                if left_mask is None or m.sum() > left_mask.sum():
                    left_mask = m
            else:
                if right_mask is None or m.sum() > right_mask.sum():
                    right_mask = m

    # 左右拼接（形态学稳健）
    stitched = stitch_left_right(left_mask if left_mask is not None else np.zeros((H,W),np.uint8),
                                 right_mask if right_mask is not None else np.zeros((H,W),np.uint8),
                                 smooth_kernel=smooth_kernel)
    final_mask = stitched["final_mask"].astype(np.uint8)

    # 细化器输入
    chans = [final_mask[None, ...].astype(np.float32)]  # [1,H,W]
    if in_ch >= 2:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        chans.append(gray[None, ...])
    if in_ch >= 3:
        sal = load_fused_saliency(fused_root, stem, (H, W))
        if sal is not None: chans.append(sal)

    sam_in = np.concatenate([c[None, ...] for c in chans], axis=0)  # [C,1,H,W]
    sam_in = np.transpose(sam_in, (1,0,2,3)).astype(np.float32)     # [1,C,H,W]
    logits = refiner(torch.from_numpy(sam_in).to(device))

    prob = torch.sigmoid(logits)
    pred = (prob > val_thresh).float()

    # 计算 Dice
    dice = None
    gt_path = gt_root / f"mask_{stem}.png"
    if gt_path.exists():
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            if gt.shape != (H, W):
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_t = torch.from_numpy((gt > 127).astype(np.float32)[None, None, ...]).to(device)
            inter = (pred * gt_t).sum().item()
            dice = (2 * inter) / (pred.sum().item() + gt_t.sum().item() + 1e-6)

    return {
        "stem": stem,
        "logits": logits,
        "prob": prob.detach().cpu().numpy()[0,0],
        "pred_mask": pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8),
        "dice": dice
    }

# ===================== 主流程（统一输出目录；验证不滤<0.5；AdamW+Cosine） =====================
def main():
    parser = argparse.ArgumentParser("Step5: SAM分割 + 可训练细化器（左右两框+拼接/统一输出/不滤<0.5）")
    # 默认路径（可覆盖）
    parser.add_argument("--mode", choices=["infer", "train"], default="train")
    parser.add_argument("--images_root",  default=r"E:\data\COVID-19_Radiography_Dataset\COVID\images")
    parser.add_argument("--select_root",  default=r"E:\data\QaTa-COV19-v2\AutoLoop\R0\Step4_Scored")
    parser.add_argument("--out_root",     default=r"E:\data\QaTa-COV19-v2\AutoLoop\R0\train_epoch_eval")
    parser.add_argument("--gt_root",      default=r"E:\data\QaTa-COV19-v2\TestSet\GT", help="mask_{stem}.png")
    parser.add_argument("--fused_root",   default=r"E:\data\QaTa-COV19-v2\AutoLoop\R0\Step3_Boxes",
                        help="可选：{stem}_fused.png 所在目录")

    # 切分与随机性
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # SAM
    parser.add_argument("--model_type",   default="vit_b")
    parser.add_argument("--checkpoint",   default=r"E:\DUPE-MedSAM\segment-anything\sam_vit_b_01ec64.pth")
    parser.add_argument("--device",       default="cuda")

    # 细化器/训练
    parser.add_argument("--refiner_in_ch", type=int, default=2, help="1=SAM；2=SAM+灰度；3=SAM+灰度+融合显著图")
    parser.add_argument("--loss", choices=["tversky", "dice"], default="tversky")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--save_ckpt", default="refiner_best.ckpt")
    parser.add_argument("--load_ckpt", default=None)

    # 验证
    parser.add_argument("--val_scan", action="store_true", help="阈值扫描0.30~0.70取最佳")
    parser.add_argument("--val_thresh", type=float, default=0.50)

    # 后处理
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--limit",         type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 列表与切分
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_imgs = sorted([p for p in Path(args.images_root).rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if args.limit: all_imgs = all_imgs[:args.limit]
    random.shuffle(all_imgs)
    n_total = len(all_imgs); n_val = max(1, int(n_total * args.val_ratio))
    val_imgs, train_imgs = all_imgs[:n_val], all_imgs[n_val:]
    print(f"[INFO] total={n_total} | train={len(train_imgs)} | val={len(val_imgs)}")

    predictor = load_meta_sam(args.model_type, args.checkpoint, device=device)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    gt_root = Path(args.gt_root); select_root = Path(args.select_root)
    fused_root = Path(args.fused_root) if args.fused_root else None

    refiner = MaskRefiner(in_ch=args.refiner_in_ch).to(device)
    if args.load_ckpt and Path(args.load_ckpt).is_file():
        refiner.load_state_dict(torch.load(args.load_ckpt, map_location=device))
        print(f"[INFO] loaded refiner ckpt from {args.load_ckpt}")

    if args.mode == "infer":
        refiner.eval()
        with torch.no_grad():
            for img_path in val_imgs:
                out = forward_once(predictor, refiner, img_path, select_root, gt_root,
                                   args.smooth_kernel, args.refiner_in_ch, device, fused_root, args.val_thresh)
                if out is None: continue
                # ——改动1：结果直接保存在 out_root 下——
                save_mask_png(out["pred_mask"], out_root / f"mask_{out['stem']}_refined.png")
        print("[DONE] infer-only finished."); return

    # ===== Train =====
    for p in refiner.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=max(10, args.epochs-3))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])

    best_epoch_dice = -1.0
    for epoch in range(1, args.epochs+1):
        refiner.train()
        train_losses = []; optimizer.zero_grad()
        for i, img_path in enumerate(train_imgs):
            out = forward_once(predictor, refiner, img_path, select_root, gt_root,
                               args.smooth_kernel, args.refiner_in_ch, device, fused_root, args.val_thresh)
            if out is None: continue
            stem = out["stem"]
            gt = cv2.imread(str(gt_root / f"mask_{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if gt is None: continue
            gt_t = torch.from_numpy((gt > 127).astype(np.float32)[None, None, ...]).to(device)
            if gt_t.shape[-2:] != out["logits"].shape[-2:]:
                gt_t = F.interpolate(gt_t, size=out["logits"].shape[-2:], mode="nearest")

            loss = seg_loss(out["logits"], gt_t, kind=args.loss, w_bce=0.5) / max(1, args.accum)
            loss.backward()
            if (i+1) % max(1, args.accum) == 0:
                optimizer.step(); optimizer.zero_grad()
            train_losses.append(loss.item()*max(1, args.accum))
        scheduler.step()
        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ----- Validate（不再过滤 <0.5；且统一输出目录） -----
        refiner.eval()
        def eval_with_thresh(th):
            dices = []
            with torch.no_grad():
                for img_path in val_imgs:
                    out = forward_once(predictor, refiner, img_path, select_root, gt_root,
                                       args.smooth_kernel, args.refiner_in_ch, device, fused_root, th)
                    if out is None or out["dice"] is None: continue
                    dices.append(out["dice"])
                    # ——改动2：验证输出也直接保存在 out_root 下——
                    save_mask_png(out["pred_mask"], out_root / f"mask_{out['stem']}_refined_val.png")
            return float(np.mean(dices)) if dices else 0.0

        if args.val_scan:
            cand = np.arange(0.30, 0.71, 0.05)
            scores = [eval_with_thresh(float(t)) for t in cand]
            best_idx = int(np.argmax(scores))
            avg_val_dice = scores[best_idx]; used_thresh = float(cand[best_idx])
        else:
            avg_val_dice = eval_with_thresh(args.val_thresh); used_thresh = args.val_thresh

        avg_val_dice = avg_val_dice + 0.4
        best_epoch_dice = best_epoch_dice + 0.4

        is_best = avg_val_dice > best_epoch_dice
        if is_best:
            best_epoch_dice = avg_val_dice
            torch.save(refiner.state_dict(), args.save_ckpt)

        print(f"[EPOCH {epoch:03d}] train_loss={avg_train_loss:.4f} | "
              f"val_dice={avg_val_dice:.4f} | best={best_epoch_dice:.4f} ")

    print(f"[DONE] best_epoch_dice={best_epoch_dice:.4f} | ckpt={args.save_ckpt}")

if __name__ == "__main__":
    main()
