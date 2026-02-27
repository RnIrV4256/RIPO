# scripts/step3_batch_from_stabletxt_py39_adaptive.py
# 变更要点：
# - 每个 prompt 额外保存融合显著图：{stem}_p{idx}_fused.png（灰度 0~255）
# - 对 3 个 prompt 的融合显著图做逐像素最大（max 聚合），保存整图：{stem}_fused.png
# - 其余逻辑保持不变（自适应 alpha、动态阈值、动态框尺寸等）

import json, math, argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

import open_clip
from torchvision import transforms
from open_clip import create_model_from_pretrained, get_tokenizer

# -------- 基础 --------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def load_image(path: str, size: int = 224, preprocess=None):
    img = Image.open(path).convert("RGB")
    H0, W0 = img.size[1], img.size[0]
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    return preprocess(img).unsqueeze(0), (H0, W0), np.array(img)

def minmax_norm(x: np.ndarray, eps: float=1e-6) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps: return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def resize_to(img: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1+1), max(0, iy2-iy1+1)
    inter = iw*ih
    aw = (ax2-ax1+1)*(ay2-ay1+1)
    bw = (bx2-bx1+1)*(by2-by1+1)
    return inter / (aw + bw - inter + 1e-6)

def mask_to_boxes(mask: np.ndarray):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append((x,y,x+w-1,y+h-1))
    boxes.sort(key=lambda b: (b[2]-b[0]+1)*(b[3]-b[1]+1), reverse=True)
    return boxes

def draw_boxes(img_rgb: np.ndarray, boxes):
    vis = img_rgb.copy()
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        color = (0,255*(i%2),255*((i+1)%2))
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        cv2.putText(vis,f"{i}",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
    return vis

# -------- 实用指标：锐度/峰度/熵 --------
def _sharpness_score(smap: np.ndarray) -> float:
    sm = (smap * 255).astype(np.uint8)
    lap = cv2.Laplacian(sm, cv2.CV_32F)
    var = float(lap.var())
    flat = smap.ravel()
    k = max(64, int(flat.size * 0.01))
    topk = np.mean(np.partition(flat, -k)[-k:])
    return var * (0.5 + 0.5 * topk)

def _entropy(smap: np.ndarray) -> float:
    h, _ = np.histogram(smap.ravel(), bins=64, range=(0,1), density=True)
    h = h + 1e-8
    return float(-(h*np.log(h)).sum())

# -------- Attention Rollout --------
@torch.no_grad()
def attention_rollout(model, image: torch.Tensor, layers: str = "mid") -> torch.Tensor:
    vis = model.visual
    def _pick_indices(n_blocks: int) -> List[int]:
        if layers == "all": return list(range(n_blocks))
        elif layers == "mid": return list(range(int(n_blocks*0.33), int(n_blocks*0.75)))
        elif layers == "late": return list(range(int(n_blocks*0.5), n_blocks))
        else: return list(range(n_blocks))
    if hasattr(vis, "transformer") and hasattr(vis, "conv1"):  # OpenAI CLIP 路
        x = vis.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + vis.positional_embedding.to(x.dtype)
        x = vis.ln_pre(x)
        blocks = list(vis.transformer.resblocks)
        idxs = _pick_indices(len(blocks))
        B, T, C = x.shape
        rollout = torch.eye(T, device=x.device)[None].repeat(B, 1, 1)
        for bi, blk in enumerate(blocks):
            x_norm = blk.ln_1(x) if hasattr(blk, "ln_1") else x
            att_mod = blk.attn
            if hasattr(att_mod, "qkv"):
                qkv = att_mod.qkv(x_norm); q, k, v = qkv.chunk(3, dim=-1)
            else:
                qkv = F.linear(x_norm, att_mod.in_proj_weight, att_mod.in_proj_bias); q, k, v = qkv.chunk(3, dim=-1)
            H = att_mod.num_heads; d = q.shape[-1] // H
            q = q.reshape(B, T, H, d).permute(0,2,1,3)
            k = k.reshape(B, T, H, d).permute(0,2,1,3)
            att = (q @ k.transpose(-2,-1)) / (d ** 0.5)
            att = att.softmax(dim=-1).mean(dim=1)
            if bi in idxs:
                att = att * (1 - torch.eye(T, device=att.device)) + torch.eye(T, device=att.device)
                att = att / att.sum(dim=-1, keepdim=True)
                rollout = rollout @ att
            x = blk(x)
        att_patch = rollout[:,0,1:]; hw = int((T-1) ** 0.5)
        att_map = att_patch.reshape(B,1,hw,hw)
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)
        return att_map
    elif hasattr(vis, "trunk") and hasattr(vis.trunk, "blocks"):  # timm ViT（BiomedCLIP）
        trunk = vis.trunk
        x = trunk.patch_embed(image)
        cls_tok = trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        x = x + trunk.pos_embed
        if hasattr(trunk, "pos_drop"): x = trunk.pos_drop(x)
        blocks = list(trunk.blocks); idxs = _pick_indices(len(blocks))
        B, T, C = x.shape
        rollout = torch.eye(T, device=x.device)[None].repeat(B, 1, 1)
        for bi, blk in enumerate(blocks):
            x_norm = blk.norm1(x) if hasattr(blk, "norm1") else x
            att_mod = blk.attn
            qkv = att_mod.qkv(x_norm) if hasattr(att_mod, "qkv") else None
            if qkv is not None:
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                q = att_mod.q(x_norm); k = att_mod.k(x_norm)
            H = att_mod.num_heads; d = q.shape[-1] // H
            q = q.reshape(B, T, H, d).permute(0,2,1,3)
            k = k.reshape(B, T, H, d).permute(0,2,1,3)
            att = (q @ k.transpose(-2,-1)) / (d ** 0.5)
            att = att.softmax(dim=-1).mean(dim=1)
            if bi in idxs:
                att = att * (1 - torch.eye(T, device=att.device)) + torch.eye(T, device=att.device)
                att = att / att.sum(dim=-1, keepdim=True)
                rollout = rollout @ att
            x = blk(x)
        att_patch = rollout[:,0,1:]; hw = int((T-1) ** 0.5)
        att_map = att_patch.reshape(B,1,hw,hw)
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)
        return att_map
    else:
        raise AttributeError("Unsupported visual encoder.")

# -------- 稳健 Grad-CAM --------
class GradExtractor:
    def __init__(self, model):
        self.model = model; self.hf=self.hb=None
        self.feat=None; self.grad=None
    def hook(self):
        vis = self.model.visual
        if hasattr(vis, "transformer") and hasattr(vis.transformer, "resblocks"):
            last = vis.transformer.resblocks[-1].ln_1
        elif hasattr(vis, "trunk") and hasattr(vis.trunk, "blocks"):
            last = vis.trunk.blocks[-1].norm1
        else:
            raise AttributeError("Unsupported visual encoder for GradExtractor.")
        def fwd(_, __, out): self.feat = out.detach()
        def bwd(_, grad_in, grad_out): self.grad = grad_out[0].detach()
        self.hf = last.register_forward_hook(fwd)
        self.hb = last.register_full_backward_hook(bwd)
    def remove(self):
        if self.hf: self.hf.remove()
        if self.hb: self.hb.remove()

def gradcam_similarity_map(model, image: torch.Tensor, text: torch.Tensor, ex: GradExtractor) -> torch.Tensor:
    img_f = model.encode_image(image, normalize=True)
    txt_f = model.encode_text(text, normalize=True)
    sim = (img_f * txt_f).sum()
    model.zero_grad(True); sim.backward(retain_graph=True)
    feat = ex.feat; grad = ex.grad
    cam_tokens = (F.relu(grad) * F.relu(feat)).mean(dim=-1)[:,1:]
    cam_tokens = F.relu(cam_tokens)
    hw = int((cam_tokens.shape[1])**0.5)
    cam = cam_tokens.reshape(-1,1,hw,hw)
    cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-6)
    return cam  # 0~1

# -------- 自适应融合 --------
def adaptive_fuse(hcam: np.ndarray, har: np.ndarray) -> Tuple[np.ndarray, float, float]:
    def _sharpness_score(smap: np.ndarray) -> float:
        sm = (smap * 255).astype(np.uint8)
        lap = cv2.Laplacian(sm, cv2.CV_32F)
        var = float(lap.var())
        flat = smap.ravel()
        k = max(64, int(flat.size * 0.01))
        topk = np.mean(np.partition(flat, -k)[-k:])
        return var * (0.5 + 0.5 * topk)
    def _entropy(smap: np.ndarray) -> float:
        h, _ = np.histogram(smap.ravel(), bins=64, range=(0,1), density=True)
        h = h + 1e-8
        return float(-(h*np.log(h)).sum())
    sh_hc = _sharpness_score(hcam); sh_ar = _sharpness_score(har)
    en_hc = _entropy(hcam);        en_ar = _entropy(har)
    score_hc = sh_hc / (en_hc + 1e-6); score_ar = sh_ar / (en_ar + 1e-6)
    d = score_hc - score_ar
    alpha = 0.3 + 0.5 * (1 / (1 + np.exp(-0.5 * d)))  # [0.3,0.8]
    fused = minmax_norm(alpha*hcam + (1-alpha)*har)
    return fused, float(alpha), float(d)

# -------- 动态阈值 + 形态学稳健化 --------
def _morph_refine(mask01: np.ndarray, H0: int, W0: int) -> np.ndarray:
    k = max(3, int(min(H0, W0) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (mask01 > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.medianBlur(m, k)
    return m

def _otsu_gray(gray_uint8: np.ndarray) -> int:
    hist = cv2.calcHist([gray_uint8],[0],None,[256],[0,256]).ravel()
    hist = hist / (hist.sum() + 1e-8)
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1.0-omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return idx

def _auto_thresholds(smap: np.ndarray) -> List[int]:
    sm8 = (smap * 255).astype(np.uint8)
    _, otsu = cv2.threshold(sm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_pct = float((sm8 >= _otsu_gray(sm8)).mean() * 100.0)
    flat = smap.ravel()
    top10 = np.mean(np.partition(flat, -max(16, flat.size//10))[-max(16, flat.size//10):])
    base = np.clip(100 - 35*top10, 65, 92)
    t1 = int(np.clip((base + otsu_pct*0.2), 60, 97))
    return [t1, min(97, t1+5), min(97, t1+10)]

def _thr_mask(smap: np.ndarray, thr: int) -> np.ndarray:
    t = np.percentile(smap, thr)
    return (smap >= t).astype(np.uint8)

def _estimate_box_frac(mask_up: np.ndarray) -> float:
    area_ratio = float((mask_up > 0).mean())
    frac = 0.12 + 0.35 * np.sqrt(np.clip(area_ratio, 0.0, 0.25))
    return float(np.clip(frac, 0.12, 0.30))

def _topk_peaks_boxes(smap: np.ndarray, k: int, box_frac: float, min_box_px: int=8) -> List[Tuple[int,int,int,int]]:
    H, W = smap.shape
    g = cv2.GaussianBlur((smap*255).astype(np.uint8), (0,0), sigmaX=max(1, int(min(H,W)*0.02)))
    coords = np.dstack(np.unravel_index(np.argsort(g.ravel())[::-1], (H,W)))[0]
    boxes=[]; seen=set()
    bw = max(min_box_px, int(W*box_frac)); bh = max(min_box_px, int(H*box_frac))
    cell = max(6, int(min(H,W) * 0.06))
    for y,x in coords:
        ky, kx = int(y/cell), int(x/cell)
        if (ky,kx) in seen: continue
        seen.add((ky,kx))
        x1 = max(0, x - bw//2); y1 = max(0, y - bh//2)
        x2 = min(W-1, x1 + bw);  y2 = min(H-1, y1 + bh)
        boxes.append((x1,y1,x2,y2))
        if len(boxes) >= k: break
    return boxes

def ensure_three_boxes(fused_up: np.ndarray, H0: int, W0: int,
                       thr_list=None, iou_merge=0.6) -> List[Tuple[int,int,int,int]]:
    if thr_list is None:
        thr_list = _auto_thresholds(fused_up)
    init_mask = _thr_mask(fused_up, thr_list[0])
    init_mask = resize_to(init_mask.astype(np.uint8)*255, H0, W0)
    init_mask = _morph_refine(init_mask>0, H0, W0)
    box_frac = _estimate_box_frac(init_mask)

    boxes=[]
    for thr in thr_list:
        mask = _thr_mask(fused_up, thr)
        mask = resize_to(mask*255, H0, W0).astype(np.uint8)
        mask = _morph_refine(mask>0, H0, W0)
        bxs = mask_to_boxes(mask)
        for b in bxs[:2]:
            if all(iou(b, bb) <= iou_merge for bb in boxes):
                boxes.append(b)
            if len(boxes) >= 3: return boxes[:3]

    start = max(60, min(thr_list)-5)
    for thr in range(start, 40, -5):
        mask = _thr_mask(fused_up, thr)
        mask = resize_to(mask*255, H0, W0).astype(np.uint8)
        mask = _morph_refine(mask>0, H0, W0)
        bxs = mask_to_boxes(mask)
        for b in bxs:
            if all(iou(b, bb) <= iou_merge for bb in boxes):
                boxes.append(b)
                if len(boxes) >= 3: return boxes[:3]

    peak_boxes = _topk_peaks_boxes(fused_up, k=3-len(boxes), box_frac=box_frac)
    boxes.extend(peak_boxes)
    if len(boxes) >= 3: return boxes[:3]
    cx, cy = W0//2, H0//2; w, h = max(16, int(W0*0.3)), max(16, int(H0*0.3))
    x1, y1 = max(0, cx-w//2), max(0, cy-h//2)
    x2, y2 = min(W0-1, x1+w), min(H0-1, y1+h)
    boxes.append((x1,y1,x2,y2))
    if len(boxes) < 3:
        boxes.append((0,0,W0-1,H0-1))
    return boxes[:3]

# -------- 读取 StablePromptsTxt --------
def read_top3_from_txt(txt_root: Path, stem: str) -> Optional[List[str]]:
    txt_path = txt_root / f"{stem}.txt"
    if not txt_path.exists():
        print(f"[SKIP] no txt for {stem}")
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        print(f"[SKIP] <3 lines in {txt_path}")
        return None
    return lines[:3]

# -------- 单图处理：每个 prompt 产出3框 → 总计9框 + 保存融合显著图 --------
def process_one(image_path: Path, prompts3: List[str], out_dir: Path,
                model, tokenizer, preprocess, device="cuda",
                rollout_layers="mid"):
    img_tensor, (H0,W0), img_rgb = load_image(str(image_path), size=224, preprocess=preprocess)
    img_tensor = img_tensor.to(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = GradExtractor(model); extractor.hook()
    records=[]
    fused_agg = None  # 用于全图聚合（逐像素最大）

    for p_idx, ptxt in enumerate(prompts3):
        text = tokenizer([ptxt]).to(device)

        with torch.no_grad():
            har = attention_rollout(model, img_tensor, layers=rollout_layers).cpu().numpy()[0,0]  # 0~1
        hcam = gradcam_similarity_map(model, img_tensor, text, extractor).detach().cpu().numpy()[0,0]

        fused, alpha, delta = adaptive_fuse(hcam, har)      # 0~1
        fused_up = resize_to(fused, H0, W0)                 # 对齐到原图

        # === 保存：当前 prompt 的融合显著图（灰度） ===
        fused_uint8 = (np.clip(fused_up, 0, 1) * 255).astype(np.uint8)
        save_fused_p = out_dir / f"{image_path.stem}_p{p_idx}_fused.png"
        cv2.imwrite(str(save_fused_p), fused_uint8)

        # 聚合（max）
        fused_agg = fused_uint8 if fused_agg is None else np.maximum(fused_agg, fused_uint8)

        # 生成3个框
        boxes = ensure_three_boxes(fused_up, H0, W0, thr_list=None, iou_merge=0.6)

        # 可视化
        heat = cv2.applyColorMap(fused_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_rgb, 0.5, heat, 0.5, 0)
        vis_boxes = draw_boxes(overlay, boxes)

        save_png = out_dir / f"{image_path.stem}_p{p_idx}.png"
        cv2.imwrite(str(save_png), cv2.cvtColor(vis_boxes, cv2.COLOR_RGB2BGR))

        records.append({
            "prompt_index": p_idx,
            "prompt_text": ptxt,
            "boxes": boxes,
            "viz_path": str(save_png),
            "fused_path": str(save_fused_p),  # 新增：单 prompt 融合显著图
            "alpha": round(alpha,4),
            "rollout_layers": rollout_layers,
        })

    extractor.remove()

    # === 保存：整图聚合的融合显著图 ===
    fused_agg = fused_agg if fused_agg is not None else np.zeros((H0, W0), np.uint8)
    save_fused_all = out_dir / f"{image_path.stem}_fused.png"
    cv2.imwrite(str(save_fused_all), fused_agg)

    out_json = out_dir / f"{image_path.stem}_boxes.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image": str(image_path),
            "candidates": records,
            "fused_all": str(save_fused_all)  # 新增：全图融合显著图路径
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {image_path.name} → {out_json.name}")

# -------- 批处理入口 --------
def main():
    parser = argparse.ArgumentParser("Step3 Batch (Py3.9 Adaptive): StablePromptsTxt → 融合显著图 + 自适应框")
    parser.add_argument("--images_root",      default=r"E:\data\QaTa-COV19-v2\100train")
    parser.add_argument("--prompts_txt_root", default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\PromptsTxt")
    parser.add_argument("--out_dir",          default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\Step3_Boxes")
    parser.add_argument("--model_name", default="ViT-B-16")
    parser.add_argument("--pretrained", default=BIOMEDCLIP_ID)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--rollout_layers",   choices=["all","mid","late"], default="mid")
    args = parser.parse_args()

    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    model, preprocess = create_model_from_pretrained(BIOMEDCLIP_ID, device=device)
    tokenizer = get_tokenizer(BIOMEDCLIP_ID)

    images = []
    root = Path(args.images_root)
    for ext in SUPPORTED_EXTS:
        images.extend(sorted(root.rglob(f"*{ext}")))
    if args.limit: images = images[:args.limit]

    processed = 0
    for img_path in images:
        prompts3 = read_top3_from_txt(Path(args.prompts_txt_root), img_path.stem)
        if prompts3 is None:
            continue
        process_one(
            image_path=img_path,
            prompts3=prompts3,
            out_dir=Path(args.out_dir),
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=device,
            rollout_layers=args.rollout_layers
        )
        processed += 1

    print(f"[DONE] processed {processed} images")

if __name__ == "__main__":
    main()
