import numpy as np

def bio_taxo(img):
    return [
        float(np.mean(img)),
        float(np.std(img)),
        float(np.min(img)),
        float(np.max(img)),
        float(np.median(img))
    ]
