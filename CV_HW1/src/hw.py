from timeit import default_timer as timer
import glob,os
import json
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def generate_greyscale_lookup(bins):
    bin_dct = {}
    for i in range(256):
        bin_dct[i] = bins * i // 256
    return bin_dct


def generate_rgb_lookup(bins):
    bin_dct = {}
    for i in range(256):
        for j in range(256):
            for k in range(256):
                bin_dct[(i, j, k)] = (bins * i // 256, bins * j // 256, bins * k // 256)
    return bin_dct


def generate_gradient_idx(bins, val):
    return np.math.ceil((bins - 1) * val / (2*np.pi)) + (bins - 1) // 2


def extract_greyscale_histogram(img,bin_dict, bins):
    tot_pix = img.shape[0] * img.shape[1]
    h = np.zeros(bins)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h[bin_dict[img[i][j]]] += 1 / tot_pix
    return h


def extract_rgb_histogram(img, bin_dict):
    dct = {}
    tot_pix = img.shape[0] * img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            key = tuple(img[i, j])
            idx = bin_dict[key]
            if idx not in dct:
                dct[idx] = 0
            else:
                dct[idx] += 1 / tot_pix
    return dct


def extract_grad_histogram(img,bins):
    horizontal = 1 / 3 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical = 1 / 3 * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_h = signal.convolve2d(img, horizontal)
    grad_v = signal.convolve2d(img, vertical)
    h = np.zeros(bins)
    total_mag = 0
    for i in range(grad_h.shape[0]):
        for j in range(grad_h.shape[1]):
            angle = np.math.atan2(grad_v[i][j],grad_h[i][j])
            idx = generate_gradient_idx(bins,angle)
            mag = np.math.sqrt(grad_h[i][j] ** 2 + grad_v[i][j] ** 2)
            h[idx] += mag
            total_mag += mag
    return list(map(lambda x: x/total_mag,h))


def load_db(gs_bins=0,rgb_bins=0):
    os.chdir('test_dir')
    for name in glob.glob('*.jpg'):
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        print(img)
def main():
    load_db()
    # Load images
    img = cv2.imread('ex.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('ex.jpg', cv2.IMREAD_COLOR)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Create lookup tables
    bin_gs = generate_greyscale_lookup(256)
    bin_rgb = generate_rgb_lookup(256)
    start = timer()
    # Create histograms
    gs_histogram = extract_greyscale_histogram(img,bin_gs,256)
    rgb_histogram = extract_rgb_histogram(img2,bin_rgb)
    grad_histogram = extract_grad_histogram(img,256)
    end = timer()
    print(end - start)



    pass
    '''
    plt.bar(range(255), hst)
    plt.show()
    '''


if __name__ == '__main__':
    main()
