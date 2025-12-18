import numpy as np

def bio_taxo(img):
    h, w = img.shape
    return [
        np.mean(img),
        np.std(img),
        np.min(img),
        np.max(img),
        np.median(img)
    ]
