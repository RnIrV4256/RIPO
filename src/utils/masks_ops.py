import numpy as np

def binary_iou(a,b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / (union + 1e-6)