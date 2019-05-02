import os
import sys
from timeit import default_timer as timer
import cv2
import numpy as np
from scipy import signal
from scipy.spatial import distance

import matplotlib.pyplot as plt


def usage():
    sys.stderr.write("""Please see the contents of main function for usage""")
    sys.exit(1)


# Calculates Euclidean distance between histograms
def compare_euclidean(hist1, hist2):
    subimg = len(hist1)
    dist = 0
    # Get euclidean distance of each sub image
    for i in range(subimg):
        dist += distance.euclidean(hist1[i], hist2[i])
    # Take the squared of the distance so that we can combine different type of histograms for distance calculation
    return (dist / subimg) ** 2


# Calculates chi-square distance between histograms
def compare_chi_sqr(hist1, hist2):
    subimg = len(hist1)
    dist = np.zeros(subimg)
    for s in range(subimg):
        # Relative comparison- to be divided by squared distance
        b = hist1[s] + hist2[s]
        # Get the squared distance
        d = (hist1[s] - hist2[s]) ** 2
        for i in range(d.shape[0]):
            if b[i]:
                d[i] /= b[i]
        dist[s] = sum(d)
    dist /= subimg
    return sum(dist)


def experimental_greyscale(img, num_bins, grid_level):
    # The number the width and height is divided
    divide = 2 ** (grid_level - 1)
    # Number of subimages
    num_of_subimg = divide ** 2
    # For histogram normalization
    pixel_divide = img.shape[0] * img.shape[1] // num_of_subimg
    # Split the image vertically
    v_split = np.vsplit(img, divide)
    step = 256 / num_bins
    # Create histogram of all zeros initially
    hist = np.zeros((num_of_subimg, num_bins))
    # Create bins
    bins = np.linspace(step, 256, num_bins)
    for x_idx, ar in enumerate(v_split):
        # Split  the vertically split image horizontally
        hor = np.hsplit(ar, divide)
        for y_idx, v in enumerate(hor):
            # 1D array of greyscale values
            values = v.flatten()
            # Assign indices  w.r.t pixel magnitudes
            digitized = np.digitize(values, bins)
            ones = np.ones(len(digitized))
            # Add 1 to each corresponding bin
            np.add.at(hist[x_idx * divide + y_idx], digitized, ones)
    return hist / pixel_divide


def experimental_rgb(img, num_bins, grid_level):
    # Same as above
    divide = 2 ** (grid_level - 1)
    num_of_subimg = divide ** 2
    pixel_divide = img.shape[0] * img.shape[1] // num_of_subimg
    v_split = np.vsplit(img, divide)
    step = 256 / num_bins
    # Create histogram with num_of_subimgs x num_bins x num_bins x num_bins
    hist = np.zeros((num_of_subimg, num_bins, num_bins, num_bins))
    bins = np.linspace(step, 256, num_bins)
    for x_idx, ar in enumerate(v_split):
        hor = np.hsplit(ar, divide)
        for y_idx, v in enumerate(hor):
            values = v.reshape((v.shape[0] * v.shape[1], v.shape[2]))
            sp = np.hsplit(values, 3)
            # 3 different channels for R,G and B
            red = sp[0].flatten()
            green = sp[1].flatten()
            blue = sp[2].flatten()
            digitized_r = np.digitize(red, bins)
            digitized_g = np.digitize(green, bins)
            digitized_b = np.digitize(blue, bins)
            # Add 1 to each corresponding RGB triplet in the histogram
            np.add.at(hist, (x_idx * divide + y_idx, digitized_r, digitized_g, digitized_b), 1)
    return hist.reshape(hist.shape[0], hist.shape[1] * hist.shape[2] * hist.shape[3]) / pixel_divide


def experimental_gradient(img, num_bins, grid_level):
    # Vertical and horizontal gradient operators
    vertical = 1 / 3 * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    horizontal = 1 / 3 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # Same cases as mentioned above
    divide = 2 ** (grid_level - 1)
    num_of_subimg = divide ** 2
    v_split = np.vsplit(img, divide)
    hist = np.zeros((num_of_subimg, num_bins))
    # Bins go from 0 to 2pi
    bins = np.linspace(0, 2 * np.pi, num_bins)
    for x_idx, ar in enumerate(v_split):
        hor = np.hsplit(ar, divide)
        for y_idx, v in enumerate(hor):
            # Code for visualizing the gradients, not needed
            '''
            grad_h = signal.convolve2d(v, horizontal)
            grad_v = signal.convolve2d(v, vertical)
            fig = plt.figure()

            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.set_xlabel("Gx")

            ax2.set_xlabel("Gy")
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(grad_h, cmap='gray')
            ax2.imshow(grad_v, cmap='gray')
           

            plt.show()
            '''
            # Apply convolutions
            grad_h = signal.convolve2d(v, horizontal).flatten()
            grad_v = signal.convolve2d(v, vertical).flatten()
            # Get the gradient matrix
            grads = np.arctan2(grad_v, grad_h)
            # Normalize the gradient orientations to be between 0 and 2pi
            grads[grads < 0] += 2 * np.pi
            digitized = np.digitize(grads, bins, True)
            # Add the magnitudes to each corresponding bin where bins are quantized according to orientations
            mag = np.sqrt(np.square(grad_h) + np.square(grad_v))
            np.add.at(hist[x_idx * divide + y_idx], digitized, mag)

    for i in range(num_of_subimg):
        s = sum(hist[i])
        if s > 0:
            hist[i] /= s
    return hist


# Extract features(histogram data) from the dataset
def load_db(gs_bins=None, rgb_bins=None, gradient_bins=None, grid_levels=None):
    db_greyscale = {}
    db_rgb = {}
    db_gradient = {}
    gs_level, rgb_level, grad_level = grid_levels
    cnt = 0
    # Get image names from images.dat file
    with open('images.dat') as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    os.chdir('dataset')
    for name in names:

        start = timer()
        # Load images
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(name, cv2.IMREAD_COLOR)
        # In case they aren't 640 x 480 or 480 x 640
        if img.shape[0] == 640 and img.shape[1] != 480:
            img = cv2.resize(img, (img.shape[0], 480))
            img2 = cv2.resize(img2, (img.shape[0], 480))
        elif img.shape[1] == 640 and img.shape[0] != 480:
            img = cv2.resize(img, (480, img.shape[1]))
            img2 = cv2.resize(img2, (480, img2.shape[1]))
        # Create histograms
        if gs_bins:
            gs_histogram = experimental_greyscale(img, gs_bins, gs_level)
            db_greyscale[name] = gs_histogram
        if rgb_bins:
            rgb_histogram = experimental_rgb(img2, rgb_bins, rgb_level)
            db_rgb[name] = rgb_histogram
        if gradient_bins:
            grad_histogram = experimental_gradient(img, gradient_bins, grad_level)
            db_gradient[name] = grad_histogram
        end = timer()
        print('Cnt:', cnt, 'Time:', end - start)
        cnt += 1
    return db_greyscale, db_rgb, db_gradient


def generate_features(gs_bin, rgb_bin, grad_bin, grid_levels=None, write=True):
    start = timer()
    # Get gs,rgb,gradient histograms
    db_gs, db_rgb, db_gradient = load_db(gs_bin, rgb_bin, grad_bin, grid_levels)
    gs_level, rgb_level, grad_level = grid_levels
    os.chdir('..')
    # Save them by default
    if write:
        if db_gs:
            np.savez('ftrs_gs%d_gl%d' % (gs_bin, gs_level), **db_gs)
        if db_rgb:
            np.savez('ftrs_rgb%d_gl%d' % (rgb_bin, rgb_level), **db_rgb)
        if db_gradient:
            np.savez('ftrs_grad%d_gl%d' % (grad_bin, grad_level), **db_gradient)
    else:
        return (db_gs, db_rgb, db_gradient)
    end = timer()
    print('Total:', end - start)


def cbir(db, names, category, gs_bins=None, rgb_bins=None, grad_bins=None, grid_levels=None, file=True, weights=None,
         chi=False):
    # If weights are explicitly given, for combining purposes
    if weights:
        weight_gs, weight_rgb, weight_grad = weights
    else:
        weight_gs = 6
        weight_rgb = 20
        weight_grad = 58
    # Grid levels for each type of histogram
    gs_level, rgb_level, grad_level = grid_levels
    # Histogram databases for each type
    db_gs, db_rgb, db_grad = db
    count = 0
    # The string we will be writing to .out file
    file_str = ''
    # Switch to dataset directory
    os.chdir('dataset')
    for name in names:
        print(count)
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        img_clr = cv2.imread(name, cv2.IMREAD_COLOR)
        if img.shape[0] == 640 and img.shape[1] != 480:
            img = cv2.resize(img, (img.shape[0], 480))
            img_clr = cv2.resize(img_clr, (img_clr.shape[0], 480))
        elif img.shape[1] == 640 and img.shape[0] != 480:
            img = cv2.resize(img, (480, img.shape[1]))
            img_clr = cv2.resize(img_clr, (480, img_clr.shape[1]))
        greyscale_histogram = None
        rgb_histogram = None
        gradient_histogram = None
        valid = None
        if db_gs:
            greyscale_histogram = experimental_greyscale(img, gs_bins, gs_level)
            valid = db_gs
        if db_rgb:
            rgb_histogram = experimental_rgb(img_clr, rgb_bins, rgb_level)
            valid = db_rgb
        if db_grad:
            gradient_histogram = experimental_gradient(img, grad_bins, grad_level)
            valid = db_grad
        dct = {}
        for key, v in valid.items():
            if key == name:
                continue
            dst = 0
            # Calculate distances for each type of histogram for a single query then sum them w.r.t to their weights
            if db_rgb:
                # Each type of histogram does Euclidean comparison by default as indicated in the text
                if chi:
                    dst += weight_rgb * compare_chi_sqr(rgb_histogram, db_rgb[key])
                else:
                    dst += weight_rgb * compare_euclidean(rgb_histogram, db_rgb[key])
            if db_gs:
                if chi:
                    dst += weight_gs * compare_chi_sqr(greyscale_histogram, db_gs[key])
                else:
                    dst += weight_gs * compare_euclidean(greyscale_histogram, db_gs[key])
            if db_grad:
                if chi:
                    dst += weight_grad * compare_chi_sqr(gradient_histogram, db_grad[key])
                else:
                    dst += weight_grad * compare_euclidean(gradient_histogram, db_grad[key])
            # Take the square root of the combined result
            dct[key] = np.sqrt(dst)
        dct = {k: v for k, v in sorted(dct.items(), key=lambda x: x[1])}
        file_str += name + ':' + ' '.join('{} {}'.format(v, k) for k, v in dct.items()) + '\n'
        count += 1
    if file:
        os.chdir('..')
        if category == 'gs':
            with open('result_gs%d_gl%d.out' % (gs_bins, gs_level), 'w') as f:
                f.write(file_str)
        elif category == 'rgb':
            with open('result_rgb%d_gl%d.out' % (rgb_bins, rgb_level), 'w') as f:
                f.write(file_str)
        elif category == 'grad':
            with open('result_grad%d_gl%d.out' % (grad_bins, grad_level), 'w') as f:
                f.write(file_str)
        else:
            with open('result_%d_%d_%d_gl%d_%d_%d_w%d_%d_%d.out' % (
                    gs_bins, rgb_bins, grad_bins, gs_level, rgb_level, grad_level, weight_gs, weight_rgb, weight_grad),
                      'w') as f:
                f.write(file_str)
    else:
        os.chdir('..')
        with open('result.out', 'w') as f:
            f.write(file_str)


# Function that initialises  Content based image retieval(CBIR) with correct parameters
def start_cbir(gs_bins=None, rgb_bins=None, grad_bins=None, grid_levels=None, file=True, db=None, weights=None,
               chi=False):
    # Name of the query file to be read from, change it to test_queries or something else if you'd like
    with open('validation_queries.dat') as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    # For loading the intermediate results from file
    if file:
        rgb_db = {}
        gs_db = {}
        grad_db = {}
        dbgs = None
        dbrgb = None
        dbgrad = None
        gs_level, rgb_level, grad_level = grid_levels
        if gs_bins:
            dbgs = np.load('ftrs_gs%d_gl%d.npz' % (gs_bins, gs_level))
        if rgb_bins:
            dbrgb = np.load('ftrs_rgb%d_gl%d.npz' % (rgb_bins, rgb_level))
        if grad_bins:
            dbgrad = np.load('ftrs_grad%d_gl%d.npz' % (grad_bins, grad_level))
        st = timer()
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
        cbir((gs_db, rgb_db, grad_db), names, category, gs_bins, rgb_bins, grad_bins, grid_levels)
    else:
        # For the best configuration
        cbir(db, names, 'all', gs_bins, rgb_bins, grad_bins, grid_levels, False, weights, chi)


def init_extraction():
    if sys.argv[2] == 'rgb':
        generate_features(None, int(sys.argv[3]), None, (None, int(sys.argv[4]), None))
    elif sys.argv[2] == 'grad':
        generate_features(None, None, int(sys.argv[3]), (None, None, int(sys.argv[4])))
    else:
        generate_features(int(sys.argv[3]), None, None, (int(sys.argv[4]), None, None))


def main():
    if len(sys.argv) == 1:
        usage()
    '''
        Usage:
        python3 hw.py best-configuration : Gives the best preconfigured configuration with Euclidean dist.
        python3 hw.py config-chisqr: Gives the best configuration with Chi-Squared distance
        python3 hw.py extract gs 16 2 : Extracts greyscale histograms with 16 bin size and 2 grid level.
        gs can be rgb or grad for different type of histograms.
        python3 hw.py rgb 8 1: Processes the extracted rgb features with 8 bin size and grid level 1.
        Parameters can be changed as above.
        python3 hw.py all 16 16 45 1 2 3: Processes greyscale, rgb, gradient histograms with the bin sizes
        and grid levels respectively.
    '''
    # Pre configured best combinations
    if sys.argv[1] == 'best-configuration':
        db = generate_features(16, 8, 90, (3, 2, 3), False)
        start_cbir(16, 8, 90, (3, 2, 3), False, db, (3, 21, 58))
    elif sys.argv[1] == 'config-chisqr':
        db = generate_features(16, 16, 45, (3, 2, 3), False)
        start_cbir(16, 16, 45, (3, 2, 3), False, db, (6, 20, 58), True)
    # For extracting the features
    elif sys.argv[1] == 'extract':
        init_extraction()
    # For loading the saved files
    elif sys.argv[1] == 'gs':
        start_cbir(int(sys.argv[2]), None, None, (int(sys.argv[3]), None, None))
    elif sys.argv[1] == 'rgb':
        start_cbir(None, int(sys.argv[2]), None, (None, int(sys.argv[3]), None))
    elif sys.argv[1] == 'grad':
        start_cbir(None, None, int(sys.argv[2]), (None, None, int(sys.argv[3])))
    else:
        start_cbir(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                   (int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])))


if __name__ == '__main__':
    main()
