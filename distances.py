import numpy as np

def euclidienne(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def manhattan(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.abs(a - b))

def chebychev(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.max(np.abs(a - b))

def canberra_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b)
    mask = den != 0
    return np.sum(num[mask] / den[mask])

def Recherche_Image_Similaire(signatures, dist, query, k):
    res = []
    for s in signatures:
        feat = s[:-2]
        label = s[-2]
        path = s[-1]
        if dist == "euclidienne":
            d = euclidienne(feat, query)
        elif dist == "manhattan":
            d = manhattan(feat, query)
        elif dist == "chebychev":
            d = chebychev(feat, query)
        else:
            d = canberra_dist(feat, query)
        res.append((path, d, label))
    res.sort(key=lambda x: x[1])
    return res[:k]
