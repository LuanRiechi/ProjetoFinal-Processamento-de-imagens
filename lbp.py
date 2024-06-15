import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(image):
    print("Extraindo LBP features...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    print("LBP features extraídas com sucesso")
    return hist
