import os
import numpy as np
from descripteurs import concat_feat

data = []

for cls in os.listdir("dataset"):
    cls_path = os.path.join("dataset", cls)
    for f in os.listdir(cls_path):
        p = os.path.join(cls_path, f)
        feat = concat_feat(p)
        data.append(feat + [cls, os.path.join(cls, f)])

np.save("Signatures_Concat.npy", np.array(data, dtype=object))
