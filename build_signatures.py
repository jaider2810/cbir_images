import os
import numpy as np
from descripteurs import glcm_gray, haralick_feat, bit_feat, concat_feat

def extract(path, desc):
    if desc == "glcm":
        return glcm_gray(path)
    if desc == "haralick":
        return haralick_feat(path)
    if desc == "bit":
        return bit_feat(path)
    return concat_feat(path)

def build(dataset_root, descriptor, out):
    data = []
    for cls in os.listdir(dataset_root):
        cls_path = os.path.join(dataset_root, cls)
        for f in os.listdir(cls_path):
            img_path = os.path.join(cls_path, f)
            feat = extract(img_path, descriptor)
            rel = os.path.join(cls, f)
            data.append(feat + [cls, rel])
    np.save(out, np.array(data, dtype=object))

if __name__ == "__main__":
    build("dataset", "concat", "Signatures_Concat.npy")
