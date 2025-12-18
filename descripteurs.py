import cv2
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo

def glcm_gray(path):
    img = cv2.imread(path, 0)
    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    return [
        float(graycoprops(glcm, 'contrast')[0,0]),
        float(graycoprops(glcm, 'homogeneity')[0,0]),
        float(graycoprops(glcm, 'correlation')[0,0]),
        float(graycoprops(glcm, 'energy')[0,0]),
        float(graycoprops(glcm, 'ASM')[0,0])
    ]

def haralick_feat(path):
    img = cv2.imread(path, 0)
    return haralick(img).mean(axis=0).tolist()

def bit_feat(path):
    img = cv2.imread(path, 0)
    return bio_taxo(img)

def concat_feat(path):
    return glcm_gray(path) + haralick_feat(path) + bit_feat(path)
