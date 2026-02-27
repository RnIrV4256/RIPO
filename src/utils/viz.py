import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont


def draw_box(im, b, color=(255,0,0)):
    x1,y1,x2,y2 = b
    im = im.copy()
    cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
    return im


def save_debug_panels(image_path, best_candidate, mask, out_path):
    im = np.array(Image.open(image_path).convert('RGB'))
    b = best_candidate["box_up"]
    im_box = draw_box(im, b, (0,255,0))

    H = best_candidate["H"]
    Hc = (255 * (H - H.min()) / (H.max()-H.min()+1e-6)).astype(np.uint8)
    Hc = cv2.applyColorMap(Hc, cv2.COLORMAP_JET)
    Hc = cv2.resize(Hc, (im.shape[1], im.shape[0]))

    # overlay mask
    m3 = np.dstack([mask*255, np.zeros_like(mask), np.zeros_like(mask)]).astype(np.uint8)
    overlay = cv2.addWeighted(im, 1.0, m3, 0.4, 0)

    top = np.concatenate([im, im_box], axis=1)
    bottom = np.concatenate([Hc, overlay], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    Image.fromarray(panel).save(out_path)