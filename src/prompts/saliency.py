import numpy as np
import cv2
import torch
import open_clip
from PIL import Image

__OC_CACHE = {"model": None, "preprocess": None, "tokenizer": None, "device": "cpu", "cfg": None}


def _auto_device(dev):
    if dev and dev != "auto":
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_openclip(cfg_openclip: dict):
    """Load OpenCLIP once and cache. cfg_openclip keys: model_name, pretrained, device"""
    global __OC_CACHE
    if __OC_CACHE["model"] is not None:
        return __OC_CACHE
    model_name = cfg_openclip.get("model_name", "RN50")
    pretrained = cfg_openclip.get("pretrained", "openai")
    device = _auto_device(cfg_openclip.get("device", "auto"))

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)

    __OC_CACHE.update({
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "device": device,
        "cfg": cfg_openclip,
    })
    return __OC_CACHE


def load_and_resize(image_path, short_side=352):
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    scale = short_side / min(w, h)
    new_w, new_h = int(w*scale), int(h*scale)
    im_small = im.resize((new_w, new_h), Image.BILINEAR)
    return np.array(im_small), {"scale": scale, "orig_wh": (w, h)}


# ---------- Grad-CAM for OpenCLIP RN50 ---------- #
class _FeatGrad:
    def __init__(self):
        self.feats = []
        self.grads = []
    def fwd(self, m, i, o):
        self.feats.append(o.detach())
    def bwd(self, m, gi, go):
        self.grads.append(go[0].detach())


def _gradcam_rn50(model, preprocess, tokenizer, device, img_np, text: str):
    # hooks on layer4 for spatial features
    target_layer = model.visual.layer4
    hook = _FeatGrad()
    h1 = target_layer.register_forward_hook(hook.fwd)
    h2 = target_layer.register_full_backward_hook(hook.bwd)

    try:
        img_t = preprocess(Image.fromarray(img_np)).unsqueeze(0).to(device)
        txt_t = tokenizer([text]).to(device)
        # forward
        image_features = model.encode_image(img_t)
        text_features = model.encode_text(txt_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())[0, 0]
        # backward
        model.zero_grad(set_to_none=True)
        logits.backward(retain_graph=True)
        A = hook.feats[-1][0]   # (C,H,W)
        G = hook.grads[-1][0]   # (C,H,W)
        weights = G.mean(dim=(1,2))
        cam = torch.relu((weights[:, None, None] * A).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        cam_np = cam.detach().cpu().numpy()
        H, W = img_np.shape[:2]
        cam_np = cv2.resize(cam_np, (W, H), interpolation=cv2.INTER_CUBIC)
        return cam_np
    finally:
        h1.remove(); h2.remove()


# ---------- Attention Rollout fallback (ViT) ---------- #
@torch.no_grad()
def _attn_rollout_vit(model, preprocess, tokenizer, device, img_np, text: str, head_fuse: str = "mean"):
    # Simple rollout via CLS attention in last blocks, using timm hooks that expose attn in outputs
    # Not all ViT in open-clip expose raw attn; this is a light-weight fallback using token-grad magnitude.
    img_t = preprocess(Image.fromarray(img_np)).unsqueeze(0).to(device)
    txt_t = tokenizer([text]).to(device)
    # encode to get gradients of image tokens magnitude wrt similarity
    img_t.requires_grad_(True)
    image_features = model.encode_image(img_t)
    text_features = model.encode_text(txt_t)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits = (logit_scale * image_features @ text_features.t())[0,0]
    model.zero_grad(set_to_none=True)
    logits.backward()
    # use input gradients as a saliency proxy
    grad = img_t.grad.detach()[0]  # (3,H,W) model input space
    sal = grad.abs().mean(dim=0).cpu().numpy()
    sal = (sal - sal.min())/(sal.max()-sal.min()+1e-6)
    H, W = img_np.shape[:2]
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    return sal


def get_saliency(image_path, text, short_side=352, cfg_openclip: dict = None):
    """Public API used by pipeline: returns (saliency H, img_small, resize_meta)."""
    img_small, meta = load_and_resize(image_path, short_side)
    H = saliency_from_np(img_small, text, cfg_openclip)
    return H, img_small, meta


def saliency_from_np(img_np: np.ndarray, text: str, cfg_openclip: dict = None):
    cache = init_openclip(cfg_openclip or {})
    model = cache["model"]; preprocess = cache["preprocess"]; tokenizer = cache["tokenizer"]; device = cache["device"]
    # choose path by model family
    name = (cache.get("cfg", {}) or {}).get("model_name", "RN50").upper()
    try:
        if name.startswith("RN"):
            H = _gradcam_rn50(model, preprocess, tokenizer, device, img_np, text)
        else:
            H = _attn_rollout_vit(model, preprocess, tokenizer, device, img_np, text)
    except Exception:
        # robust fallback: grayscale center-bias (should be rare)
        g = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = (g - g.min())/(g.max()-g.min()+1e-6)
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        cy, cx = g.shape[0]/2, g.shape[1]/2
        R = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        Rn = R / (R.max()+1e-6)
        H = (1.0 - Rn) * g
        H = (H - H.min())/(H.max()-H.min()+1e-6)
    return H

import numpy as np
import cv2
from PIL import Image

# TODO: replace with real CLIP forward + Grad-CAM / Attention Rollout.
# Here we return a dummy saliency: a blurred intensity map emphasizing center.

def load_and_resize(image_path, short_side=352):
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    scale = short_side / min(w, h)
    new_w, new_h = int(w*scale), int(h*scale)
    im_small = im.resize((new_w, new_h), Image.BILINEAR)
    return np.array(im_small), {"scale": scale, "orig_wh": (w, h)}


def get_saliency(image_path, text, short_side=352):
    img, meta = load_and_resize(image_path, short_side)
    H = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H = H / (H.max() + 1e-6)
    H = cv2.GaussianBlur(H, (3,3), 0)
    # emphasize central area as a placeholder
    yy, xx = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    cy, cx = H.shape[0]/2, H.shape[1]/2
    R = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    Rn = R / R.max()
    H = (1.0 - Rn) * H
    H = (H - H.min()) / (H.max() - H.min() + 1e-6)
    return H, img, meta