# scripts/step5_sam_infer_and_post.py
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2

import torch

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

# ===================== 多连通后处理 =====================
def fill_holes(binary: np.ndarray) -> np.ndarray:
    """洪泛填洞，输入0/1；输出0/1"""
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    im = (binary > 0).astype(np.uint8).copy()
    flood = im.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood = (flood == 0).astype(np.uint8)
    return (im | flood).astype(np.uint8)

def postprocess_multi_component(
    masks: List[np.ndarray],
    img_shape: Tuple[int, int],
    min_area: int = 2000,
    keep_per_side: int = 1,
    do_split_lr: bool = True,
    smooth_kernel: int = 7,
) -> Dict:
    """
    合并多个掩码 → 形态学清理/填洞 → 连通域分析 → 按左右半侧各保留N块（肺默认为1）。
    返回：{final_mask, components[{area, bbox, centroid, side}]}
    """
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

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser("Step5: SAM分割 + 多连通后处理（最优prompt的3个框→3个mask→最终肺掩码）")
    # 路径（Windows 友好：直接改默认值即可双击运行）
    parser.add_argument("--images_root",  default=r"E:\data\QaTa-COV19-v2\100train")
    parser.add_argument("--select_root",  default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\Step4_Scored")
    parser.add_argument("--out_root",     default=r"E:\data\QaTa-COV19-v2\Step5_SAM")
    # SAM配置
    parser.add_argument("--model_type",   default="vit_b", help="vit_h | vit_l | vit_b")
    parser.add_argument("--checkpoint",   default=r"E:\DUPE-MedSAM\segment-anything\sam_vit_b_01ec64.pth")
    parser.add_argument("--device",       default="cuda")
    # 后处理参数
    parser.add_argument("--min_area",     type=int, default=2000, help="连通域最小面积（像素）")
    parser.add_argument("--keep_per_side",type=int, default=1,    help="每侧保留连通域数：肺=1，病灶可>1")
    parser.add_argument("--do_split_lr",  action="store_true", help="按左右半侧分组（肺推荐开启）")
    parser.add_argument("--smooth_kernel",type=int, default=7)
    parser.add_argument("--limit",        type=int, default=None)
    args = parser.parse_args()

    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    # 枚举图像（大小写无关）
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    img_paths = sorted([p for p in Path(args.images_root).rglob("*")
                        if p.is_file() and p.suffix.lower() in exts])
    if args.limit: img_paths = img_paths[:args.limit]
    print(f"[INFO] found {len(img_paths)} images")

    # 加载SAM
    predictor = load_meta_sam(args.model_type, args.checkpoint, device=device)

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    for img_path in img_paths:
        stem = img_path.stem
        sel = load_select_json(Path(args.select_root), stem)
        if not sel or "boxes_for_sam" not in sel:
            print(f"[SKIP] no select json / boxes_for_sam for {stem}")
            continue
        boxes = [tuple(map(int, b)) for b in sel["boxes_for_sam"]]  # 3个框

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[SKIP] cannot read {img_path}")
            continue

        # 逐框 SAM 推理
        masks = run_meta_sam_for_boxes(predictor, img_bgr, boxes)

        # 保存单掩码与叠加图
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(masks):
            save_mask_png(m, out_dir / f"{stem}_mask_{i}.png")

        vis = img_bgr.copy()
        colors = [(0,255,0), (0,165,255), (255,0,0)]
        for i, m in enumerate(masks):
            vis = overlay_mask(vis, m, color=colors[i%len(colors)], alpha=0.45)
            x1,y1,x2,y2 = boxes[i]
            cv2.rectangle(vis,(x1,y1),(x2,y2),colors[i%len(colors)],2)
            cv2.putText(vis, f"box{i}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i%len(colors)], 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{stem}_sam_overlay.png"), vis)

        # 多连通后处理（肺：左右各保留1块；小噪声剔除；填洞；平滑）
        H, W = img_bgr.shape[:2]
        adaptive_min_area = max(args.min_area, int(0.002 * H * W))  # 自适应下限（二者取大）
        info = postprocess_multi_component(
            masks=masks,
            img_shape=(H, W),
            min_area=adaptive_min_area,
            keep_per_side=args.keep_per_side,
            do_split_lr=args.do_split_lr,
            smooth_kernel=args.smooth_kernel,
        )
        final_mask = info["final_mask"]
        save_mask_png(final_mask, out_dir / f"{stem}_final_mask.png")

        # 最终叠加
        final_vis = overlay_mask(img_bgr, final_mask, color=(0,255,0), alpha=0.45)
        cv2.imwrite(str(out_dir / f"{stem}_final_overlay.png"), final_vis)

        # 汇总 JSON
        out_json = {
            "image": str(img_path),
            "best_prompt_index": sel.get("best_prompt_index"),
            "best_prompt_text":  sel.get("best_prompt_text"),
            "boxes_for_sam": boxes,
            "mask_paths": [str(out_dir / f"{stem}_mask_{i}.png") for i in range(len(masks))],
            "overlay_paths": str(out_dir / f"{stem}_sam_overlay.png"),
            "final_mask": str(out_dir / f"{stem}_final_mask.png"),
            "final_overlay": str(out_dir / f"{stem}_final_overlay.png"),
            "components": info["components"],  # 每个保留连通域：area/bbox/centroid/side
            "params": {
                "min_area": adaptive_min_area,
                "keep_per_side": args.keep_per_side,
                "do_split_lr": args.do_split_lr,
                "smooth_kernel": args.smooth_kernel,
                "model_type": args.model_type
            }
        }
        (out_root / f"{stem}_final.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

    print(f"[DONE] SAM masks + multi-component postprocess finished for {processed} images")
    try: input("\n任务完成，按回车退出...")
    except: pass

if __name__ == "__main__":
    main()
