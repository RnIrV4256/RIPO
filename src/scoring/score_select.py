# scripts/step4_select_prompt_and_boxes.py
import math, json, argparse, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip

# ---------------- 基础工具 ----------------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def draw_boxes(img_rgb: np.ndarray, boxes: List[Tuple[int,int,int,int]], color=(0,255,0)):
    vis = img_rgb.copy()
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        cv2.putText(vis,f"{i}",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)
    return vis

def minmax_norm(x: np.ndarray, eps: float=1e-6) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps: return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    a = (a>0).astype(np.uint8); b = (b>0).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum() + 1e-6
    return float(inter) / float(union)

def box_mask(shape: Tuple[int,int], box: Tuple[int,int,int,int]) -> np.ndarray:
    H, W = shape
    x1,y1,x2,y2 = box
    m = np.zeros((H,W), np.uint8)
    x1 = max(0,min(W-1,x1)); x2 = max(0,min(W-1,x2))
    y1 = max(0,min(H-1,y1)); y2 = max(0,min(H-1,y2))
    m[y1:y2+1, x1:x2+1] = 1
    return m

# ---------------- 读取 Step2/Step3 结果 ----------------
def read_top3_from_txt(txt_root: Path, stem: str) -> Optional[List[str]]:
    p = txt_root / f"{stem}.txt"
    if not p.exists(): return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 3: return None
    return lines[:3]

def read_boxes_json(boxes_root: Path, stem: str) -> Optional[List[List[Tuple[int,int,int,int]]]]:
    p = boxes_root / f"{stem}_boxes.json"
    if not p.exists(): return None
    data = json.loads(p.read_text(encoding="utf-8"))
    cands = data.get("candidates", [])
    if len(cands) < 3: return None
    per_prompt = []
    for it in cands[:3]:
        boxes = it.get("boxes", [])
        if not boxes: return None
        per_prompt.append([tuple(map(int, b)) for b in boxes][:3])  # 3个框
    return per_prompt

# ---------------- CLIP & 显著性 ----------------
def load_image_tensor(path: str, size: int = 224):
    img = Image.open(path).convert("RGB")
    H0, W0 = img.size[1], img.size[0]
    preprocess = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return preprocess(img).unsqueeze(0), (H0,W0), np.array(img)

@torch.no_grad()
def attention_rollout(model, image: torch.Tensor) -> torch.Tensor:
    """兼容 open_clip.Attention / nn.MultiheadAttention 的 Rollout"""
    visual = model.visual
    assert hasattr(visual, "transformer")
    x = visual.conv1(image)                                    # [B,C,h,w]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)   # [B,N,C]
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],1,x.shape[-1], device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    B,T,C = x.shape
    rollout = torch.eye(T, device=x.device).unsqueeze(0).repeat(B,1,1)

    for blk in visual.transformer.resblocks:
        x_norm = blk.ln_1(x) if hasattr(blk,"ln_1") else x
        att_mod = blk.attn
        if hasattr(att_mod, "qkv"):
            qkv = att_mod.qkv(x_norm); q,k,v = qkv.chunk(3, dim=-1)
        elif hasattr(att_mod, "in_proj_weight"):
            qkv = F.linear(x_norm, att_mod.in_proj_weight, att_mod.in_proj_bias); q,k,v = qkv.chunk(3, dim=-1)
        else:
            raise RuntimeError("Unsupported attention module")
        H = att_mod.num_heads
        d = q.shape[-1] // H
        q = q.reshape(B,T,H,d).permute(0,2,1,3)
        k = k.reshape(B,T,H,d).permute(0,2,1,3)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(d)          # [B,H,T,T]
        att = att.softmax(dim=-1).mean(dim=1)                  # [B,T,T]
        att = att * (1 - torch.eye(T, device=att.device)) + torch.eye(T, device=att.device)
        att = att / att.sum(dim=-1, keepdim=True)
        rollout = rollout @ att
        x = blk(x)                                             # 用 block 整体前向

    att_patch = rollout[:,0,1:]
    hw = int((T-1)**0.5)
    att_map = att_patch.reshape(B,1,hw,hw)
    att_map = (att_map - att_map.min()) / (att_map.max()-att_map.min()+1e-6)
    return att_map

class GradExtractor:
    def __init__(self, model): self.model=model; self.hf=self.hb=None; self.feat=None; self.grad=None
    def hook(self):
        last = self.model.visual.transformer.resblocks[-1].ln_1
        def fwd(_, __, out): self.feat = out.detach()
        def bwd(_, grad_in, grad_out): self.grad = grad_out[0].detach()
        self.hf = last.register_forward_hook(fwd)
        self.hb = last.register_full_backward_hook(bwd)
    def remove(self):
        if self.hf: self.hf.remove()
        if self.hb: self.hb.remove()

def gradcam_similarity_map(model, image: torch.Tensor, text: torch.Tensor, ex: GradExtractor) -> torch.Tensor:
    img_f = model.encode_image(image); txt_f = model.encode_text(text)
    img_f = img_f/img_f.norm(dim=-1,keepdim=True); txt_f = txt_f/txt_f.norm(dim=-1,keepdim=True)
    sim = (img_f*txt_f).sum(); model.zero_grad(True); sim.backward(retain_graph=True)
    feat,grad = ex.feat, ex.grad; assert feat is not None and grad is not None
    w = grad.mean(dim=1, keepdim=True)
    cam_tokens = (feat*w).sum(dim=-1)[:,1:]
    cam_tokens = F.relu(cam_tokens)
    hw = int((cam_tokens.shape[1])**0.5)
    cam = cam_tokens.reshape(-1,1,hw,hw)
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-6)
    return cam

def fuse_saliency(hcam: np.ndarray, har: np.ndarray, alpha: float=0.6) -> np.ndarray:
    return minmax_norm(alpha*hcam + (1-alpha)*har)

# ---------------- 扰动（不改几何） ----------------
def photometric_perturb(img_rgb: np.ndarray) -> np.ndarray:
    out = img_rgb.astype(np.float32) / 255.0
    b = random.uniform(-0.06, 0.06)       # 亮度
    c = random.uniform(0.92, 1.08)        # 对比度
    out = np.clip(out * c + b, 0, 1)
    sigma = random.uniform(0.0, 0.02)     # 噪声
    if sigma > 0:
        out = np.clip(out + np.random.normal(0, sigma, out.shape).astype(np.float32), 0, 1)
    if random.random() < 0.5:             # 轻模糊
        k = random.choice([3,5])
        out = cv2.GaussianBlur(out, (k,k), 0)
    return (out * 255.0).astype(np.uint8)

def preprocess_tensor(img_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = Image.fromarray(img_rgb)
    pre = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return pre(img).unsqueeze(0)

# ---------------- 指标与区域 ----------------
def mask_from_fused(fused_up: np.ndarray, box: Tuple[int,int,int,int], thr_percent: int=90) -> np.ndarray:
    H,W = fused_up.shape
    bx = box_mask((H,W), box)
    # 在框内自适应阈值
    thr = np.percentile(fused_up[bx>0], thr_percent) if (bx>0).sum()>0 else np.percentile(fused_up, thr_percent)
    m = (fused_up >= thr).astype(np.uint8) & bx
    return m

def s1_consistency_iou(ref_mask: np.ndarray, pert_masks: List[np.ndarray]) -> float:
    s = [iou_mask(ref_mask, m) for m in pert_masks]
    return float(np.mean(s)) if s else 0.0

def s2_area_stability(pert_masks: List[np.ndarray]) -> float:
    areas = np.array([m.sum() for m in pert_masks], dtype=np.float32)
    if areas.mean() < 1e-6: return 0.0
    cv = areas.std() / (areas.mean() + 1e-6)
    return float(1.0 - np.clip(cv, 0.0, 1.0))

def s3_focus_consistency(fused_up: np.ndarray, box: Tuple[int,int,int,int], ring: int=6) -> float:
    x1,y1,x2,y2 = box
    H,W = fused_up.shape
    x1r = max(0, x1-ring); y1r = max(0, y1-ring)
    x2r = min(W-1, x2+ring); y2r = min(H-1, y2+ring)
    inner = fused_up[y1:y2+1, x1:x2+1]
    outer = fused_up[y1r:y2r+1, x1r:x2r+1].copy()
    outer[y1:y2+1, x1:x2+1] = np.nan
    hin  = float(np.nanmean(inner)) if inner.size else 0.0
    hout = float(np.nanmean(outer)) if np.isfinite(outer).any() else 0.0
    diff = np.clip(hin - hout, -1.0, 1.0)
    return float((diff + 1.0)/2.0)  # 映射到 [0,1]

# ---------------- 单图评分：选最优prompt，输出其3个框 ----------------
def select_prompt_and_boxes(model, tokenizer, image_path: Path, prompts3: List[str],
                            per_prompt_boxes: List[List[Tuple[int,int,int,int]]],
                            device="cuda", alpha=0.6, thr_percent=90, T=6,
                            w1=0.5, w2=0.3, w3=0.2, aggregate="mean",
                            out_dir: Optional[Path]=None) -> Dict:
    """
    aggregate: 'mean' | 'max'（对每个prompt下3个框的综合分聚合方式）
    """
    # 原图 & 张量
    img_tensor0, (H0,W0), img_rgb0 = load_image_tensor(str(image_path), size=224)
    img_tensor0 = img_tensor0.to(device)

    extractor = GradExtractor(model); extractor.hook()
    per_prompt_scores = []   # 聚合分
    per_prompt_detail = []   # 每框分

    for p_idx, ptxt in enumerate(prompts3):
        text = tokenizer([ptxt]).to(device)
        with torch.no_grad():
            har0 = attention_rollout(model, img_tensor0).cpu().numpy()[0,0]
        hcam0 = gradcam_similarity_map(model, img_tensor0, text, extractor).detach().cpu().numpy()[0,0]
        fused0 = fuse_saliency(hcam0, har0, alpha=alpha)
        fused0_up = cv2.resize(fused0, (W0, H0), interpolation=cv2.INTER_CUBIC)

        box_scores = []
        for box in per_prompt_boxes[p_idx]:
            # 参考 mask
            ref_mask = mask_from_fused(fused0_up, box, thr_percent=thr_percent)
            # 扰动 T 次
            pert_masks = []
            for _ in range(T):
                img_p = photometric_perturb(img_rgb0)
                ten = preprocess_tensor(img_p, size=224).to(device)
                with torch.no_grad():
                    har = attention_rollout(model, ten).cpu().numpy()[0,0]
                hcam = gradcam_similarity_map(model, ten, text, extractor).detach().cpu().numpy()[0,0]
                fused = fuse_saliency(hcam, har, alpha=alpha)
                fused_up = cv2.resize(fused, (W0, H0), interpolation=cv2.INTER_CUBIC)
                pert_masks.append(mask_from_fused(fused_up, box, thr_percent=thr_percent))

            S1 = s1_consistency_iou(ref_mask, pert_masks)
            S2 = s2_area_stability(pert_masks)
            S3 = s3_focus_consistency(fused0_up, box, ring=6)
            score = w1*S1 + w2*S2 + w3*S3
            box_scores.append({"box": list(map(int, box)), "S1": S1, "S2": S2, "S3": S3, "score": score})

        # prompt 聚合分
        if aggregate == "max":
            agg = max(b["score"] for b in box_scores)
        else:
            agg = float(np.mean([b["score"] for b in box_scores]))
        per_prompt_scores.append(agg)
        per_prompt_detail.append({"prompt_index": p_idx, "prompt_text": ptxt, "boxes": box_scores})

    extractor.remove()

    # 选最优 prompt
    best_pid = int(np.argmax(per_prompt_scores))
    best_prompt = prompts3[best_pid]
    best_three_boxes = [b["box"] for b in per_prompt_detail[best_pid]["boxes"]]

    # 可视化：高亮最优 prompt 的 3 个框
    save_png = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = draw_boxes(img_rgb0, [tuple(b) for b in best_three_boxes], color=(0,255,0))
        save_png = str((out_dir / f"{image_path.stem}_best_prompt.png"))
        cv2.imwrite(save_png, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return {
        "image": str(image_path),
        "aggregate": aggregate,
        "per_prompt_scores": per_prompt_scores,
        "per_prompt_detail": per_prompt_detail,
        "best_prompt_index": best_pid,
        "best_prompt_text": best_prompt,
        "boxes_for_sam": best_three_boxes,  # ★ 这3个框直接喂 SAM
        "viz_path": save_png
    }

# ---------------- 批处理入口 ----------------
def main():
    parser = argparse.ArgumentParser("Step4: 选最优 prompt，并输出其3个框用于 SAM")
    parser.add_argument("--images_root",      default=r"E:\data\QaTa-COV19-v2\100train")
    parser.add_argument("--prompts_txt_root", default=r"E:\data\QaTa-COV19-v2\StablePromptsTxt")
    parser.add_argument("--boxes_root",       default=r"E:\data\QaTa-COV19-v2\Step3_Boxes")
    parser.add_argument("--out_root",         default=r"E:\data\QaTa-COV19-v2\Step4_Scored")

    parser.add_argument("--model_name", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--pretrained", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--device",           default="cuda")

    parser.add_argument("--alpha",            type=float, default=0.6)
    parser.add_argument("--thr_percent",      type=int,   default=90)
    parser.add_argument("--T",                type=int,   default=6)
    parser.add_argument("--w1",               type=float, default=0.5)
    parser.add_argument("--w2",               type=float, default=0.3)
    parser.add_argument("--w3",               type=float, default=0.2)
    parser.add_argument("--aggregate",        type=str,   default="mean", choices=["mean","max"])
    parser.add_argument("--limit",            type=int,   default=None)
    args = parser.parse_args()

    device = args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    img_paths = list_images(Path(args.images_root))
    if args.limit: img_paths = img_paths[:args.limit]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    for img_path in img_paths:
        stem = img_path.stem
        prompts3 = read_top3_from_txt(Path(args.prompts_txt_root), stem)
        boxes3x3  = read_boxes_json(Path(args.boxes_root), stem)
        if not prompts3 or not boxes3x3:
            continue

        res = select_prompt_and_boxes(
            model=model, tokenizer=tokenizer, image_path=img_path,
            prompts3=prompts3, per_prompt_boxes=boxes3x3,
            device=device, alpha=args.alpha, thr_percent=args.thr_percent, T=args.T,
            w1=args.w1, w2=args.w2, w3=args.w3, aggregate=args.aggregate,
            out_dir=out_root
        )

        (out_root / f"{stem}_select.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

    print(f"[DONE] selected prompt+3boxes for {processed} images")
    try: input("\n任务完成，按回车退出...")
    except: pass

if __name__ == "__main__":
    main()
