import numpy as np
from scipy.spatial.distance import canberra

def euclidienne(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def manhattan(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b)))

def chebychev(a, b):
    return np.max(np.abs(np.array(a) - np.array(b)))

def canberra_dist(a, b):
    return canberra(a, b)

def Recherche_Image_Similaire(signatures, distance_name, query, k):
    res = []
    for s in signatures:
        feat = s[:-2]
        label = s[-2]
        path = s[-1]
        if distance_name == "euclidienne":
            d = euclidienne(feat, query)
        elif distance_name == "manhattan":
            d = manhattan(feat, query)
        elif distance_name == "chebychev":
            d = chebychev(feat, query)
        else:
            d = canberra_dist(feat, query)
        res.append((path, d, label))
    res.sort(key=lambda x: x[1])
    return res[:k]
