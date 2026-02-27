# scripts/step3_batch_from_stabletxt_py39.py
# 改造要点：
# 1) 直接加载 BiomedCLIP via create_model_from_pretrained + 匹配 tokenizer
# 2) GradExtractor 与 attention_rollout 同时兼容 OpenAI-CLIP ViT 与 timm ViT（BiomedCLIP用）
# 3) 保持原有 I/O、默认参数与输出结构不变

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
    """
    若提供 preprocess（来自 create_model_from_pretrained 返回），则优先用之；
    否则回退到与 CLIP 兼容的手工变换。
    返回：tensor[1,3,224,224], (H0,W0), img_rgb(np.uint8 H×W×3)
    """
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

# -------- Attention Rollout：兼容 OpenAI-CLIP 与 timm ViT --------
@torch.no_grad()
def attention_rollout(model, image: torch.Tensor) -> torch.Tensor:
    """
    返回 [B,1,h,w]，范围 0~1
    A 路：OpenAI CLIP ViT: visual.transformer.resblocks, conv1/positional_embedding/ln_pre
    B 路：timm ViT (BiomedCLIP): visual.trunk.blocks, patch_embed/pos_embed/cls_token/pos_drop
    """
    import torch.nn.functional as F
    vis = model.visual

    # ---- A) OpenAI CLIP ViT 路径 ----
    if hasattr(vis, "transformer") and hasattr(vis, "conv1"):
        x = vis.conv1(image)                                   # [B,C,h,w]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B,N,C]
        cls = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
        x = torch.cat([cls, x], dim=1)                        # [B,1+N,C]
        x = x + vis.positional_embedding.to(x.dtype)
        x = vis.ln_pre(x)
        blocks = vis.transformer.resblocks

        B, T, C = x.shape
        rollout = torch.eye(T, device=x.device)[None].repeat(B, 1, 1)

        for blk in blocks:
            x_norm = blk.ln_1(x) if hasattr(blk, "ln_1") else x
            att_mod = blk.attn
            if hasattr(att_mod, "qkv"):
                qkv = att_mod.qkv(x_norm)
                q, k, v = qkv.chunk(3, dim=-1)
            elif hasattr(att_mod, "in_proj_weight"):
                qkv = F.linear(x_norm, att_mod.in_proj_weight, att_mod.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                raise RuntimeError("Unsupported attention module")

            H = att_mod.num_heads
            d = q.shape[-1] // H
            q = q.reshape(B, T, H, d).permute(0, 2, 1, 3)
            k = k.reshape(B, T, H, d).permute(0, 2, 1, 3)
            att = (q @ k.transpose(-2, -1)) / (d ** 0.5)      # [B,H,T,T]
            att = att.softmax(dim=-1).mean(dim=1)             # [B,T,T]

            att = att * (1 - torch.eye(T, device=att.device)) + torch.eye(T, device=att.device)
            att = att / att.sum(dim=-1, keepdim=True)
            rollout = rollout @ att

            x = blk(x)  # 前向推进

        att_patch = rollout[:, 0, 1:]                         # [B,N]
        hw = int((T - 1) ** 0.5)
        att_map = att_patch.reshape(B, 1, hw, hw)
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)
        return att_map

    # ---- B) timm ViT 路径（BiomedCLIP）----
    elif hasattr(vis, "trunk") and hasattr(vis.trunk, "blocks"):
        trunk = vis.trunk
        x = trunk.patch_embed(image)                          # [B,N,C]
        cls_tok = trunk.cls_token.expand(x.shape[0], -1, -1)  # [B,1,C]
        x = torch.cat((cls_tok, x), dim=1)                    # [B,1+N,C]
        x = x + trunk.pos_embed
        if hasattr(trunk, "pos_drop"):
            x = trunk.pos_drop(x)

        B, T, C = x.shape
        rollout = torch.eye(T, device=x.device)[None].repeat(B, 1, 1)

        for blk in trunk.blocks:
            x_norm = blk.norm1(x) if hasattr(blk, "norm1") else x
            att_mod = blk.attn
            if hasattr(att_mod, "qkv"):
                qkv = att_mod.qkv(x_norm)
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                # 极少数自定义注意力降级处理
                q = att_mod.q(x_norm); k = att_mod.k(x_norm)

            H = att_mod.num_heads
            d = q.shape[-1] // H
            q = q.reshape(B, T, H, d).permute(0, 2, 1, 3)
            k = k.reshape(B, T, H, d).permute(0, 2, 1, 3)
            att = (q @ k.transpose(-2, -1)) / (d ** 0.5)
            att = att.softmax(dim=-1).mean(dim=1)

            att = att * (1 - torch.eye(T, device=att.device)) + torch.eye(T, device=att.device)
            att = att / att.sum(dim=-1, keepdim=True)
            rollout = rollout @ att

            x = blk(x)

        att_patch = rollout[:, 0, 1:]
        hw = int((T - 1) ** 0.5)
        att_map = att_patch.reshape(B, 1, hw, hw)
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)
        return att_map

    else:
        raise AttributeError("Unsupported visual encoder.")

# -------- Hcam: Grad-CAM 近似（兼容双路径） --------
class GradExtractor:
    def __init__(self, model):
        self.model = model
        self.hf=self.hb=None
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
    model.zero_grad(True)
    sim.backward(retain_graph=True)

    feat = ex.feat; grad = ex.grad
    assert feat is not None and grad is not None, "Grad hooks failed."
    w = grad.mean(dim=1, keepdim=True)            # [B,1,C]
    cam_tokens = (feat * w).sum(dim=-1)[:,1:]     # 去 CLS
    cam_tokens = F.relu(cam_tokens)
    hw = int((cam_tokens.shape[1])**0.5)
    cam = cam_tokens.reshape(-1,1,hw,hw)
    cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-6)
    return cam  # 0~1

# -------- 融合 + 固定3框/Prompt --------
def fuse_saliency(hcam: np.ndarray, har: np.ndarray, alpha: float=0.6) -> np.ndarray:
    return minmax_norm(alpha*hcam + (1-alpha)*har)

def _thr_mask(smap: np.ndarray, thr: int) -> np.ndarray:
    t = np.percentile(smap, thr)
    return (smap >= t).astype(np.uint8)

def _topk_peaks_boxes(smap: np.ndarray, k: int, box_frac: float=0.2) -> List[Tuple[int,int,int,int]]:
    """从热图Top-K峰生成固定大小的方框（占短边box_frac）。"""
    H, W = smap.shape
    g = cv2.GaussianBlur((smap*255).astype(np.uint8), (0,0), sigmaX=max(1, int(min(H,W)*0.02)))
    coords = np.dstack(np.unravel_index(np.argsort(g.ravel())[::-1], (H,W)))[0]  # y,x 排序
    boxes=[]; seen=set()
    bw = int(W*box_frac); bh = int(H*box_frac)
    bw = max(8, bw); bh = max(8, bh)
    for y,x in coords:
        # 避免密集重复
        ky, kx = int(y/8), int(x/8)
        if (ky,kx) in seen: continue
        seen.add((ky,kx))
        x1 = max(0, x - bw//2); y1 = max(0, y - bh//2)
        x2 = min(W-1, x1 + bw); y2 = min(H-1, y1 + bh)
        boxes.append((x1,y1,x2,y2))
        if len(boxes) >= k: break
    return boxes

def ensure_three_boxes(fused_up: np.ndarray, H0: int, W0: int,
                       thr_list=(85,90,95), iou_merge=0.8) -> List[Tuple[int,int,int,int]]:
    """保证返回3个框：阈值→降阈→峰值→兜底。"""
    # 1) 先按阈值拿框
    boxes=[]
    for thr in thr_list:
        mask = _thr_mask(fused_up, thr)
        mask = resize_to(mask*255, H0, W0).astype(np.uint8)
        k = max(3, int(round(min(H0,W0)*0.01))|1)
        mask = cv2.medianBlur(mask, k)
        bxs = mask_to_boxes(mask)
        for b in bxs[:1]:  # 每个阈值取1个
            # 去重
            if all(iou(b, bb) <= iou_merge for bb in boxes):
                boxes.append(b)
        if len(boxes) >= 3:
            return boxes[:3]

    # 2) 自动降阈补足
    for thr in range(min(thr_list)-5, 60, -5):  # e.g. 80,75,70...
        mask = _thr_mask(fused_up, thr)
        mask = resize_to(mask*255, H0, W0).astype(np.uint8)
        bxs = mask_to_boxes(mask)
        for b in bxs:
            if all(iou(b, bb) <= iou_merge for bb in boxes):
                boxes.append(b)
                if len(boxes) >= 3:
                    return boxes[:3]

    # 3) Top-K 峰生框
    peak_boxes = _topk_peaks_boxes(fused_up, k=3-len(boxes), box_frac=0.18)
    boxes.extend(peak_boxes)
    if len(boxes) >= 3:
        return boxes[:3]

    # 4) 兜底：中心框/全图
    cx, cy = W0//2, H0//2
    w, h = max(16, int(W0*0.3)), max(16, int(H0*0.3))
    x1, y1 = max(0, cx-w//2), max(0, cy-h//2)
    x2, y2 = min(W0-1, x1+w), min(H0-1, y1+h)
    boxes.append((x1,y1,x2,y2))
    if len(boxes) < 3:
        boxes.append((0,0,W0-1,H0-1))
    return boxes[:3]

# -------- 读取 StablePromptsTxt（Py3.9写法） --------
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

# -------- 单图处理：每个 prompt 产出3框 → 总计9框 --------
def process_one(image_path: Path, prompts3: List[str], out_dir: Path,
                model, tokenizer, preprocess, alpha=0.6, thresholds=(85,90,95), device="cuda"):
    img_tensor, (H0,W0), img_rgb = load_image(str(image_path), size=224, preprocess=preprocess)
    img_tensor = img_tensor.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = GradExtractor(model); extractor.hook()
    records=[]

    for p_idx, ptxt in enumerate(prompts3):
        text = tokenizer([ptxt]).to(device)

        with torch.no_grad():
            har = attention_rollout(model, img_tensor).cpu().numpy()[0,0]     # 0~1
        hcam = gradcam_similarity_map(model, img_tensor, text, extractor).detach().cpu().numpy()[0,0]

        fused = fuse_saliency(hcam, har, alpha=alpha)
        fused_up = resize_to(fused, H0, W0)

        # ★ 每个 prompt 固定产出 3 个框
        boxes = ensure_three_boxes(fused_up, H0, W0, thr_list=thresholds, iou_merge=0.8)

        heat = (fused_up*255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_rgb, 0.5, heat, 0.5, 0)
        vis_boxes = draw_boxes(overlay, boxes)

        save_png = out_dir / f"{image_path.stem}_p{p_idx}.png"
        cv2.imwrite(str(save_png), cv2.cvtColor(vis_boxes, cv2.COLOR_RGB2BGR))

        records.append({
            "prompt_index": p_idx,
            "prompt_text": ptxt,
            "boxes": boxes,            # 3 个 [x1,y1,x2,y2]
            "viz_path": str(save_png)
        })

    extractor.remove()

    out_json = out_dir / f"{image_path.stem}_boxes.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"image": str(image_path), "candidates": records}, f, ensure_ascii=False, indent=2)
    print(f"[OK] {image_path.name} → {out_json.name}")

# -------- 批处理入口 --------
def main():
    parser = argparse.ArgumentParser("Step3 Batch (Py3.9): StablePromptsTxt → Hcam/Har → 每prompt 3框(共9)")
    parser.add_argument("--images_root",      default=r"E:\data\QaTa-COV19-v2\100train")
    parser.add_argument("--prompts_txt_root", default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\PromptsTxt")
    parser.add_argument("--out_dir",          default=r"E:\data\QaTa-COV19-v2\AutoLoop10\R0\Step3_Boxes")
    # 下面两个参数保留以兼容旧版，但实际不再使用
    parser.add_argument("--model_name", default="ViT-B-16")
    parser.add_argument("--pretrained", default=BIOMEDCLIP_ID)
    parser.add_argument("--alpha",            type=float, default=0.6)
    parser.add_argument("--thresholds",       nargs="+", type=int, default=[85,90,95])
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--limit",            type=int, default=None)
    args = parser.parse_args()

    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    # ★ 关键：直接加载 BiomedCLIP + tokenizer + preprocess
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
            alpha=float(args.alpha),
            thresholds=tuple(args.thresholds),
            device=device
        )
        processed += 1

    print(f"[DONE] processed {processed} images")
    try: input("\n任务完成，按回车退出...")
    except: pass

if __name__ == "__main__":
    main()
