import numpy as np, cv2
from skimage import measure, morphology


def _double_threshold_mask(H, high_sigma=1.0, low_sigma=0.3):
    mu, sigma = H.mean(), H.std()
    th_h = mu + high_sigma * sigma
    th_l = mu + low_sigma * sigma
    seeds = (H >= th_h)
    lo = (H >= th_l)
    M = morphology.reconstruction(seeds.astype(np.uint8), lo.astype(np.uint8), method='dilation')
    M = M.astype(bool)
    M = morphology.binary_opening(M, morphology.disk(3))
    M = morphology.binary_closing(M, morphology.disk(3))
    M = morphology.remove_small_holes(M, area_threshold=64)
    return M.astype(np.uint8)


def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [int(x1), int(y1), int(x2), int(y2)]


def _add_margin(b, shape, frac=0.06):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    dx, dy = int(frac*(x2-x1)), int(frac*(y2-y1))
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
    x2 = min(w-1, x2 + dx); y2 = min(h-1, y2 + dy)
    return [x1,y1,x2,y2]


def boxes_from_saliency(H, min_area_ratio=0.005, max_ar=6.0, margin_frac=0.06, k_top=3):
    M = _double_threshold_mask(H)
    labels = measure.label(M, connectivity=2)
    props = measure.regionprops(labels)
    h, w = H.shape[:2]
    cand = []
    for p in props:
        area = p.area
        if area < min_area_ratio * (h*w):
            continue
        y1, x1, y2, x2 = p.bbox  # note skimage uses (min_row, min_col, max_row, max_col)
        B = [x1, y1, x2-1, y2-1]
        ar = max((x2-x1), (y2-y1)) / max(1, min((x2-x1), (y2-y1)))
        if ar > max_ar:
            continue
        B = _add_margin(B, H.shape, frac=margin_frac)
        score = H[labels==p.label].mean()
        cand.append((B, score))
    cand.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in cand[:k_top]]