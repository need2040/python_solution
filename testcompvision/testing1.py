# %%
import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# %%
def load_binary_image(image_path, crop_top=0, crop_bottom=0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image=image[crop_top:crop_bottom]
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return binary_image


