import numpy as np

def clip_box(b, w, h):
    x1,y1,x2,y2 = b
    return [max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)]

def box_area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1+1) * max(0, y2-y1+1)