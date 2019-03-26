import glob
import os
import sys
from timeit import default_timer as timer
import cv2
import numpy as np
from scipy import signal
from scipy.spatial import distance
import matplotlib.pyplot as plt


def generate_gradient_idx(bins, val):
    return np.math.ceil((bins - 1) * val / (2 * np.pi))


def extract_greyscale_histogram(img, num_bins, grid_level):
    divide = 2 ** grid_level // 2
    num_of_subimg = divide ** 2
    hist = np.zeros((num_of_subimg, num_bins))
    pixel_divide = img.shape[0] * img.shape[1] // num_of_subimg
    x_step = img.shape[0] // divide
    y_step = img.shape[1] // divide

    for x_idx in range(divide):
        x_range_start = x_idx * x_step
        x_range_end = x_range_start + x_step
        for i in range(x_range_start, x_range_end):
            for y_idx in range(divide):
                y_range_start = y_idx * y_step
                y_range_end = y_range_start + y_step
                for j in range(y_range_start, y_range_end):
                    idx = (num_bins * img[i][j]) >> 8
                    hist[divide * x_idx + y_idx][idx] += 1
    for i in range(num_of_subimg):
        hist[i] /= pixel_divide
    return hist


def extract_rgb_histogram(img, bins, grid_level):
    divide = 2 ** grid_level // 2
    num_of_subimg = divide ** 2
    hist = np.zeros((num_of_subimg, bins * bins * bins))
    pixel_divide = img.shape[0] * img.shape[1] // num_of_subimg
    x_step = img.shape[0] // divide
    y_step = img.shape[1] // divide
    for x_idx in range(divide):
        x_range_start = x_idx * x_step
        x_range_end = x_range_start + x_step
        for i in range(x_range_start, x_range_end):
            for y_idx in range(divide):
                y_range_start = y_idx * y_step
                y_range_end = y_range_start + y_step
                for j in range(y_range_start, y_range_end):
                    r, g, b = img[i][j]
                    idx_b = bins * b >> 8
                    idx_g = bins * g >> 8
                    idx_r = bins * r >> 8
                    idx = bins ** 2 * idx_r + bins * idx_g + idx_b
                    hist[divide * x_idx + y_idx][idx] += 1

    for i in range(num_of_subimg):
        hist[i] /= pixel_divide

    return hist


def extract_grad_histogram(img, bins, grid_level):
    divide = 2 ** grid_level // 2
    num_of_subimg = divide ** 2
    horizontal = 1 / 3 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # sobel_h =1/3* np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # sobel_v =1/3* np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    vertical = 1 / 3 * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_h = signal.convolve2d(img, horizontal)
    grad_v = signal.convolve2d(img, vertical)
    # sobel_x = signal.convolve2d(img, sobel_h)
    # sobel_y = signal.convolve2d(img, sobel_v)
    '''
    fig =  plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(grad_h, cmap='gray');
    ax1.set_xlabel("Gx_hor")

    ax2.set_xlabel("Gx_sobel")
    ax2.imshow(sobel_x, cmap='gray');
    plt.show()
    '''
    hist = np.zeros((num_of_subimg, bins))
    x_step = img.shape[0] // divide
    y_step = img.shape[1] // divide
    for x_idx in range(divide):
        x_range_start = x_idx * x_step
        x_range_end = x_range_start + x_step
        for i in range(x_range_start, x_range_end):
            for y_idx in range(divide):
                y_range_start = y_idx * y_step
                y_range_end = y_range_start + y_step
                for j in range(y_range_start, y_range_end):
                    angle = np.math.atan2(grad_v[i][j], grad_h[i][j])
                    if angle <= 0:
                        angle += 2 * np.pi
                    idx = generate_gradient_idx(bins, angle)
                    mag = np.math.sqrt(grad_h[i][j] ** 2 + grad_v[i][j] ** 2)
                    hist[divide * x_idx + y_idx][idx] += mag
    for i in range(num_of_subimg):
        hist[i] /= sum(hist[i])
    return hist


def load_db(gs_bins=None, rgb_bins=None, gradient_bins=None, grid_level=1):
    os.chdir('dataset')
    db_greyscale = {}
    db_rgb = {}
    db_gradient = {}
    cnt = 0
    for name in glob.glob('*.jpg'):
        # Load images
        start = timer()
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(name, cv2.IMREAD_COLOR)
        # Create histograms
        if gs_bins:
            gs_histogram = extract_greyscale_histogram(img, gs_bins, grid_level)
            db_greyscale[name] = gs_histogram
        if rgb_bins:
            rgb_histogram = extract_rgb_histogram(img2, rgb_bins, grid_level)
            db_rgb[name] = rgb_histogram
        if gradient_bins:
            grad_histogram = extract_grad_histogram(img, gradient_bins, grid_level)
            db_gradient[name] = grad_histogram
        end = timer()
        print('Cnt:', cnt, 'Time:', end - start)
        cnt += 1
    return db_greyscale, db_rgb, db_gradient


def generate_features(gs_bin, rgb_bin, grad_bin, grid_level=1):
    start = timer()
    db_gs, db_rgb, db_gradient = load_db(gs_bin, rgb_bin, grad_bin, grid_level)
    os.chdir('..')
    if db_gs:
        np.savez('ftrs_gs%d_gl%d' % (gs_bin, grid_level), **db_gs)
    if db_rgb:
        np.savez('ftrs_rgb%d_gl%d' % (rgb_bin, grid_level), **db_rgb)
    if db_gradient:
        np.savez('ftrs_grad%d_gl%d' % (grad_bin, grid_level), **db_gradient)
    end = timer()
    print('Total:', end - start)


def compare_euclidean(hist1, hist2):
    subimg = len(hist1)
    dist = 0
    for i in range(subimg):
        dist += distance.euclidean(hist1[i], hist2[i])
    return dist / subimg


def compare_chi_sqr(hist1, hist2):
    subimg = len(hist1)
    dist = np.zeros(subimg)
    for s in range(subimg):
        for i in range(hist1[0].shape[0]):
            b = hist1[s][i] + hist2[s][i]
            if b:
                a = (hist1[s][i] - hist2[s][i]) ** 2
                dist[s] += a / b
    dist /= subimg
    return sum(dist)


def cbir(db, names, category, gs_bins=None, rgb_bins=None, grad_bins=None, grid_level=1):
    db_gs, db_rgb, db_grad = db
    count = 0
    file_str = ''
    os.chdir('../dataset')
    for name in names:
        print(count)
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        img_clr = cv2.imread(name, cv2.IMREAD_COLOR)
        greyscale_histogram = None
        rgb_histogram = None
        gradient_histogram = None
        valid = None
        if db_gs:
            greyscale_histogram = extract_greyscale_histogram(img, gs_bins, grid_level)
            valid = db_gs
        if db_rgb:
            rgb_histogram = extract_rgb_histogram(img_clr, rgb_bins, grid_level)
            valid = db_rgb
        if db_grad:
            gradient_histogram = extract_grad_histogram(img, grad_bins, grid_level)
            valid = db_grad
        dct = {}
        for key, v in valid.items():
            if key == name:
                continue
            dst = 0
            if db_rgb:
                dst += compare_euclidean(rgb_histogram, db_rgb[key])
            if db_gs:
                dst += compare_euclidean(greyscale_histogram, db_gs[key])
            if db_grad:
                dst += compare_euclidean(gradient_histogram, db_grad[key])

            dct[key] = dst
        dct = {k: v for k, v in sorted(dct.items(), key=lambda x: x[1])}
        file_str += name + ':' + ' '.join('{} {}'.format(v, k) for k, v in dct.items()) + '\n'
        count += 1
    os.chdir('../results')
    if category == 'gs':
        with open('result_gs%d_gl%d.out' % (gs_bins, grid_level), 'w') as f:
            f.write(file_str)
    elif category == 'rgb':
        with open('result_rgb%d_gl%d.out' % (rgb_bins, grid_level), 'w') as f:
            f.write(file_str)
    elif category == 'grad':
        with open('result_grad%d_gl%d.out' % (grad_bins, grid_level), 'w') as f:
            f.write(file_str)
    else:
        with open('result_%d_%d_%d_gl%d.out' % (gs_bins, rgb_bins, grad_bins, grid_level), 'w') as f:
            f.write(file_str)


def start_cbir(gs_bins=None, rgb_bins=None, grad_bins=None, grid_level=1):
    with open('validation_queries.dat') as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    rgb_db = {}
    gs_db = {}
    grad_db = {}
    os.chdir('ftrs')
    dbgs = None
    dbrgb = None
    dbgrad = None
    if gs_bins:
        dbgs = np.load('ftrs_gs%d_gl%d.npz' % (gs_bins, grid_level))
    if rgb_bins:
        dbrgb = np.load('ftrs_rgb%d_gl%d.npz' % (rgb_bins, grid_level))
    if grad_bins:
        dbgrad = np.load('ftrs_grad%d_gl%d.npz' % (grad_bins, grid_level))
    # db = (), np.load('ftrs_rgb32_gl2.npz'), np.load('ftrs_grad360_gl3.npz'))
    st = timer()
    valid = None
    if dbgs:
        valid = dbgs
    elif dbrgb:
        valid = dbrgb
    else:
        valid = dbgrad
    for name in valid.files:
        if dbgs:
            gs_db[name] = dbgs[name]
        if dbrgb:
            rgb_db[name] = dbrgb[name]
        if dbgrad:
            grad_db[name] = dbgrad[name]
    ed = timer()
    print(ed - st)
    category = None
    if valid is dbgs:
        category = 'gs'
    elif valid is dbrgb:
        category = 'rgb'
    elif valid is dbgrad:
        category = 'grad'
    if (dbgs and dbrgb) and dbgrad:
        category = 'all'
    cbir((gs_db, rgb_db, grad_db), names, category, gs_bins, rgb_bins, grad_bins, grid_level)


def init_extraction():
    if sys.argv[1] == 'rgb':
        generate_features(None, int(sys.argv[2]), None, int(sys.argv[3]))
    elif sys.argv[1] == 'grad':
        generate_features(None, None, int(sys.argv[2]), int(sys.argv[3]))
    else:
        generate_features(int(sys.argv[2]), None, None, int(sys.argv[3]))


def main():
    # init_extraction()
    '''
    os.chdir('dataset')
    sample = cv2.imread('bcelcLwVNa.jpg',cv2.IMREAD_GRAYSCALE)
    sample_clr = cv2.imread('bcelcLwVNa.jpg',cv2.IMREAD_COLOR)
    my_hist = extract_grad_histogram(sample,360,1)

    print(sum(my_hist[0]))
    a = 4
    pass
     os.chdir('dataset')
    query = cv2.imread('awWzsqdQhm.jpg',cv2.IMREAD_GRAYSCALE)
    sample = cv2.imread('BjQURBXeex.jpg',cv2.IMREAD_GRAYSCALE)
    hist1 = extract_grad_histogram(query,256,3)
    hist2 = extract_grad_histogram(sample,256,3)

    diff1 = np.linalg.norm(hist1-hist2,2)
    diff2 = compare_euclidean(hist1,hist2)
    print(diff1,diff2)
    '''
    if sys.argv[1] == 'gs':
        start_cbir(int(sys.argv[2]), None, None, 1)
    elif sys.argv[1] == 'rgb':
        start_cbir(None, int(sys.argv[2]), None, 1)
    elif sys.argv[1] == 'grad':
        start_cbir(None, None, int(sys.argv[2]), 1)
    else:
        start_cbir(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))





if __name__ == '__main__':
    main()
