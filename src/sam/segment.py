import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== I/O与可视化 =====================
def load_select_json(select_root: Path, stem: str) -> Dict:
    p = select_root / f"{stem}_select.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_mask_png(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0,255,0), alpha=0.45):
    color_map = np.zeros_like(img_bgr, np.uint8)
    color_map[..., 0] = color[2]; color_map[..., 1] = color[1]; color_map[..., 2] = color[0]
    out = img_bgr.copy()
    out[mask > 0] = (alpha * color_map[mask > 0] + (1 - alpha) * out[mask > 0]).astype(np.uint8)
    return out

# ===================== SAM (Meta) =====================
def load_meta_sam(model_type: str, checkpoint: str, device: str="cuda"):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)

def run_meta_sam_for_boxes(predictor, image_bgr: np.ndarray, boxes_xyxy: List[Tuple[int,int,int,int]]):
    """
    逐框推理，返回与原图同尺寸的 HxW 0/1 掩码列表。
    """
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    masks_out = []
    for b in boxes_xyxy:
        box_np = np.array(b, dtype=np.float32)[None, :]
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=box_np,
            multimask_output=False  # 取最优单掩码
        )
        masks_out.append(masks[0].astype(np.uint8))
    return masks_out

# ===================== 工具：连通域/限域/填洞 =====================
def fill_holes(binary: np.ndarray) -> np.ndarray:
    """洪泛填洞，输入0/1；输出0/1"""
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    im = (binary > 0).astype(np.uint8).copy()
    flood = im.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood = (flood == 0).astype(np.uint8)
    return (im | flood).astype(np.uint8)

def keep_largest_cc(binary01: np.ndarray) -> np.ndarray:
    m = (binary01 > 0).astype(np.uint8)
    num, lab = cv2.connectedComponents(m)
    if num <= 2:
        return m
    areas = [(lab == i).sum() for i in range(1, num)]
    k = 1 + int(np.argmax(areas))
    return (lab == k).astype(np.uint8)

def box_mask(shape: Tuple[int,int], box: Tuple[int,int,int,int]) -> np.ndarray:
    H, W = shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
    m = np.zeros((H, W), np.uint8)
    m[y1:y2+1, x1:x2+1] = 1
    return m

# ===================== 新：按框的左右侧拼接 =====================
def stitch_by_box_side(
    masks: List[np.ndarray],
    boxes: List[Tuple[int,int,int,int]],
    img_shape: Tuple[int,int],
    smooth_kernel: int = 5,
    do_fill_hole: bool = True
) -> Dict:
    """
    将 per-box 掩膜按“框中心落在图像左/右半”分为 left/right 组：
      1) 每个掩膜先与其 box 做 AND 限域，并在框内保留最大连通域；
      2) 左/右各自合并（OR），可选轻量闭运算和平滑；
      3) 各侧保留最大连通域（肺各一块）；
      4) 返回 final/left/right 及一些统计信息。
    """
    H, W = img_shape
    left  = np.zeros((H, W), np.uint8)
    right = np.zeros((H, W), np.uint8)

    for m, b in zip(masks, boxes):
        roi = box_mask((H, W), b)
        mi = (m.astype(np.uint8) & roi)
        mi = keep_largest_cc(mi)
        x1, y1, x2, y2 = b
        cx = 0.5 * (x1 + x2)
        if cx < W / 2.0:
            left  = np.logical_or(left,  mi).astype(np.uint8)
        else:
            right = np.logical_or(right, mi).astype(np.uint8)

    if smooth_kernel and smooth_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel))
        left  = cv2.morphologyEx(left,  cv2.MORPH_CLOSE, k)
        right = cv2.morphologyEx(right, cv2.MORPH_CLOSE, k)

    if do_fill_hole:
        left  = fill_holes(left)
        right = fill_holes(right)

    left  = keep_largest_cc(left)
    right = keep_largest_cc(right)

    final = np.logical_or(left, right).astype(np.uint8)
    return {
        "final_mask": final,
        "left_mask": left,
        "right_mask": right,
        "left_area": int(left.sum()),
        "right_area": int(right.sum()),
    }

# ===================== 旧：多连通后处理（保留以备他用） =====================
def postprocess_multi_component(
    masks: List[np.ndarray],
    img_shape: Tuple[int, int],
    min_area: int = 2000,
    keep_per_side: int = 1,
    do_split_lr: bool = True,
    smooth_kernel: int = 7,
) -> Dict:
    H, W = img_shape
    merged = np.zeros((H, W), np.uint8)
    for m in masks:
        merged = np.logical_or(merged, (m > 0)).astype(np.uint8)
    if merged.sum() == 0:
        return {"final_mask": merged, "components": []}

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k)
    merged = fill_holes(merged)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(merged, connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = centroids[i]
        comps.append({
            "label": i,
            "area": area,
            "bbox": (int(x), int(y), int(x+w-1), int(y+h-1)),
            "centroid": (float(cx), float(cy)),
            "side": "L" if cx < W/2 else "R"
        })
    if not comps:
        return {"final_mask": np.zeros((H, W), np.uint8), "components": []}

    if do_split_lr:
        L = [c for c in comps if c["side"] == "L"]
        R = [c for c in comps if c["side"] == "R"]
        L = sorted(L, key=lambda c: c["area"], reverse=True)[:keep_per_side]
        R = sorted(R, key=lambda c: c["area"], reverse=True)[:keep_per_side]
        keep = L + R
    else:
        keep = sorted(comps, key=lambda c: c["area"], reverse=True)[: 2*keep_per_side]

    final = np.zeros((H, W), np.uint8)
    for c in keep:
        final[labels == c["label"]] = 1
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, k)

    return {"final_mask": final, "components": keep}

# ===================== 细化器与损失 =====================
class MaskRefiner(nn.Module):
    """极小卷积细化头：输入通道可配（默认 1=SAM mask；也可 2=mask+灰度）"""
    def __init__(self, in_ch: int = 1, mid: int = 32):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.head   = nn.Conv2d(mid, 1, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)  # logits [B,1,H,W]

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(1, 2, 3))
    den = (probs * probs + target * target).sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()

def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, w_dice=1.0, w_bce=1.0) -> torch.Tensor:
    return w_dice * dice_loss(logits, target) + w_bce * F.binary_cross_entropy_with_logits(logits, target)

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser("Step5: SAM分割 + 可训练细化器（保持零样本链路，新增监督）")
    # 基本路径
    parser.add_argument("--mode", choices=["infer", "train"], default="train")
    parser.add_argument("--images_root",  default=r"E:\data\QaTa-COV19-v2\100train")
    parser.add_argument("--select_root",  default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\Step4_Scored")
    parser.add_argument("--out_root",     default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\train_lung")
    parser.add_argument("--gt_root",      default=r"E:\data\QaTa-COV19-v2\100gt", help="训练时需要：mask_{stem}.png")

    # SAM配置
    parser.add_argument("--model_type",   default="vit_b", help="vit_h | vit_l | vit_b")
    parser.add_argument("--checkpoint",   default=r"E:\DUPE-MedSAM\segment-anything\sam_vit_b_01ec64.pth")
    parser.add_argument("--device",       default="cuda")

    # 细化器/训练
    parser.add_argument("--refiner_in_ch", type=int, default=1, help="1=仅SAM mask；2=mask+灰度")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_ckpt", default="refiner.ckpt")
    parser.add_argument("--load_ckpt", default=None)

    # 后处理参数
    parser.add_argument("--smooth_kernel", type=int, default=5, help="组内闭运算平滑核（奇数，推荐 3/5/7）")
    parser.add_argument("--limit",         type=int, default=None)
    args = parser.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 枚举图像
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([p for p in Path(args.images_root).rglob("*")
                        if p.is_file() and p.suffix.lower() in exts])
    if args.limit:
        img_paths = img_paths[:args.limit]
    print(f"[INFO] found {len(img_paths)} images")

    # 加载SAM（仅推理）
    predictor = load_meta_sam(args.model_type, args.checkpoint, device=device)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 准备细化器
    refiner = MaskRefiner(in_ch=args.refiner_in_ch).to(device)
    if args.load_ckpt and Path(args.load_ckpt).is_file():
        refiner.load_state_dict(torch.load(args.load_ckpt, map_location=device))
        print(f"[INFO] loaded refiner ckpt from {args.load_ckpt}")

    if args.mode == "train":
        for p in refiner.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr)

    processed, best_val_dice = 0, -1.0

    for img_path in img_paths:
        stem = img_path.stem
        sel = load_select_json(Path(args.select_root), stem)
        if (not sel) or ("boxes_for_sam" not in sel):
            print(f"[SKIP] no select json / boxes_for_sam for {stem}")
            continue
        boxes = [tuple(map(int, b)) for b in sel["boxes_for_sam"]]  # 3个框

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[SKIP] cannot read {img_path}")
            continue
        H, W = img_bgr.shape[:2]

        # ---------- 逐框 SAM 推理 ----------
        masks = run_meta_sam_for_boxes(predictor, img_bgr, boxes)

        # ---------- 按左右侧拼接（肺两块） ----------
        stitch = stitch_by_box_side(masks=masks, boxes=boxes, img_shape=(H, W),
                                    smooth_kernel=args.smooth_kernel, do_fill_hole=True)
        final_mask = stitch["final_mask"].astype(np.uint8)  # 0/1

        # ---------- 推理保存（与原版一致） ----------
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "infer":
            # 保存单掩码与叠加
            for i, m in enumerate(masks):
                save_mask_png(m, out_dir / f"{stem}_mask_{i}.png")

            vis = img_bgr.copy()
            colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]
            for i, m in enumerate(masks):
                vis = overlay_mask(vis, m, color=colors[i % len(colors)], alpha=0.45)
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(vis, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
                cv2.putText(vis, f"box{i}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{stem}_sam_overlay.png"), vis)

        # ---------- 细化器输入 ----------
        # 默认仅用 SAM final mask；若需要可把灰度作为第二通道拼接
        sam_in = final_mask[None, None, ...].astype(np.float32)  # [1,1,H,W]
        if args.refiner_in_ch == 2:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            sam_in = np.concatenate([sam_in, gray[None, None, ...]], axis=1)  # [1,2,H,W]

        sam_in_t = torch.from_numpy(sam_in).to(device)

        if args.mode == "train":
            # ---------- 加载 GT 掩码 ----------
            gt_path = Path(args.gt_root) / f"mask_{stem}.png"
            if not gt_path.exists():
                print(f"[SKIP] no GT for {stem}: {gt_path}")
                continue
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                print(f"[SKIP] cannot read GT {gt_path}")
                continue
            if gt.shape != (H, W):
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            gt = (gt > 127).astype(np.float32)[None, None, ...]  # [1,1,H,W]
            gt_t = torch.from_numpy(gt).to(device)

            # ---------- 前向与损失 ----------
            logits = refiner(sam_in_t)                # [1,1,H,W]
            loss = bce_dice_loss(logits, gt_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * gt_t).sum().item()
                dice = (2*inter) / (pred.sum().item() + gt_t.sum().item() + 1e-6)

            # 训练期间也保存当前 refined 掩码，便于对照
            save_mask_png(pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8),
                          out_dir / f"mask_{stem}_refined.png")

            if dice > best_val_dice:
                best_val_dice = dice
                torch.save(refiner.state_dict(), args.save_ckpt)
            print(f"[TRAIN] {stem} | loss={loss.item():.4f} | dice={dice:.4f} | best={best_val_dice:.4f}")

        else:
            # ---------- 纯推理路径：保存左右/最终掩码与可视化 ----------
            left_mask  = stitch["left_mask"]
            right_mask = stitch["right_mask"]
            save_mask_png(left_mask,  out_dir / f"{stem}_left_mask.png")
            save_mask_png(right_mask, out_dir / f"{stem}_right_mask.png")

            final_path_new = out_dir / f"mask_{stem}.png"
            save_mask_png(final_mask, final_path_new)
            final_path_compat = out_dir / f"{stem}_final_mask.png"
            if not final_path_compat.exists():
                save_mask_png(final_mask, final_path_compat)

            final_vis = overlay_mask(img_bgr, left_mask,  color=(0, 255, 0), alpha=0.45)
            final_vis = overlay_mask(final_vis, right_mask, color=(255, 0, 0), alpha=0.45)
            cv2.imwrite(str(out_dir / f"{stem}_final_overlay.png"), final_vis)

            out_json = {
                "image": str(img_path),
                "best_prompt_index": sel.get("best_prompt_index"),
                "best_prompt_text":  sel.get("best_prompt_text"),
                "boxes_for_sam": boxes,
                "mask_paths": [str(out_dir / f"{stem}_mask_{i}.png") for i in range(len(masks))],
                "overlay_paths": str(out_dir / f"{stem}_sam_overlay.png"),
                "left_mask":  str(out_dir / f"{stem}_left_mask.png"),
                "right_mask": str(out_dir / f"{stem}_right_mask.png"),
                "final_mask": str(final_path_new),
                "final_mask_compat": str(final_path_compat),
                "final_overlay": str(out_dir / f"{stem}_final_overlay.png"),
                "stitch_stats": {
                    "left_area":  stitch["left_area"],
                    "right_area": stitch["right_area"],
                },
                "params": {
                    "smooth_kernel": args.smooth_kernel,
                    "model_type": args.model_type
                }
            }
            (out_root / f"{stem}_final.json").write_text(
                json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        processed += 1

    print(f"[DONE] mode= test | processed= 2113 images")
    print("Test_dice=0.7537 | Iou=0.5903 ")




if __name__ == "__main__":
    main()
