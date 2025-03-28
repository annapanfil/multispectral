import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import blob_dog
from math import sqrt
from scipy import signal, ndimage, misc
import itertools
from scipy.stats import multivariate_normal
import pickle


def is_blob_ok(blob, img):
    h,w,c = img.shape
    x = blob[0]
    y = blob[1]
    r = blob[2]
    return  not(x - r < 0 or x + r >= w or y - r < 0 or y + r >= h)

def sigma_function1(height):
    scale = 0.15
    if height<=5:
        sigma = 6
        thresh = 0.15
    elif 5<height<=10:
        sigma = 4
        thresh = 0.1
    elif 10<height<=20:
        sigma=3
        thresh = 0.1
    # elif 20<height<=50:
    #     sigma = 2
    #     thresh = 0.05
    else:
        sigma = 2
        thresh = 0.05

    return sigma, scale, thresh

def extract_blobs3(img_gray, min_sigma=1, max_sigma=10, threshold=0.3, overlap=0.99):
    blobs_dog = blob_dog(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    keypoints = []
    for blob in blobs_dog:
        y, x, r = tuple(blob)
        keypoints += [(x, y, r)]

    return keypoints, blobs_dog

def plot(img, figsize = (8,8)):

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    plt.axis("off")
    plt.imshow(img)
    fig.tight_layout()

def filter_blobs_indices(blobs, img):
    keep_indices = []
    for i in range(len(blobs)):
        blob = blobs[i]
        if is_blob_ok(blob, img):
            keep_indices += [i]
    return np.array(keep_indices)

def get_colorspaces(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    l,a,b = cv2.split(lab_img)
    # r,g,bb = cv2.split(rgb_img)
    # h,s,v = cv2.split(hsv_img)
    # y,cr,cb = cv2.split(ycrcb_img)
    # yy,u,vv = cv2.split(yuv_img)
    #return [l,a,b,r,g,bb,h,s,v,y,cr,cb,yy,u,vv]
    return [l, a, b]

def gradient_m(i, j, picture):
    ft = (float(picture[i, j + 1]) - picture[i, j - 1]) ** 2
    st = (float(picture[i + 1, j]) - picture[i - 1, j]) ** 2

    return np.sqrt(ft + st)

def gradient_theta(i, j, picture):
    # To avoid error division by zero
    eps = 1e-5

    ft = float(picture[i + 1, j]) - picture[i - 1, j]
    st = (float(picture[i, j + 1]) - picture[i, j - 1]) + eps

    ret = 180 + np.arctan2(ft, st) * 180 / np.pi
    return ret

def create_histogram(i, j, picture, std, k_size):
    truncate = 4.0
    #kernel_size = 2 * int(std * truncate + .5) + 1
    kernel_size = k_size
    window = list(range(-kernel_size, kernel_size))
    # plt.imshow(picture[i-kernel_size:i+kernel_size, j-kernel_size:j+kernel_size])
    # plt.show()
    diag = set(itertools.permutations(window, 2))
    rooti, rootj = i, j
    theta_list = []

    gaussian = multivariate_normal(mean=[i, j], cov=1.5 * std)

    orient_hist = np.zeros([36, 1])

    for ii, jj in diag:
        x = rooti + ii
        y = rootj + jj
        if x - 1 < 0 or y - 1 < 0 or x + 1 > picture.shape[0] - 1 \
                or y + 1 > picture.shape[1] - 1:
            continue

        # TODO: Warning the magnitude are really small
        magnitude = gradient_m(x, y, picture)
        weight = magnitude * gaussian.pdf([x, y])

        orientation = gradient_theta(x, y, picture)
        bins_orientation = np.clip(orientation // 10, 0, 35)
        orient_hist[int(bins_orientation)] += weight

    return orient_hist

def transform_angle(alpha):
    if 0<alpha<180: return alpha + 180
    else: return alpha - 180

def compute_angle(img_gray, keypoint, scale=0.5, std=sqrt(2)):
    h, w = img_gray.shape
    #img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_shape = (int(h * scale), int(w * scale))
    blurred = ndimage.filters.gaussian_filter(cv2.resize(img_gray, (blur_shape[1], blur_shape[0])), std)
    kp_blur = cv2.KeyPoint(keypoint.pt[0] * scale, keypoint.pt[1] * scale, _size=keypoint.size * scale)
    orient_hist = create_histogram(int(kp_blur.pt[1]), int(kp_blur.pt[0]), blurred, std, int(kp_blur.size / 2))
    sorted_hist = np.argsort(orient_hist, axis=0)[::-1]
    angle = transform_angle(sorted_hist[0][0] * 10)
    keypoint.angle = angle

def create_kps(blob, img_gray, n_kps):
    if n_kps==3:
        kpmid = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.5)
        compute_angle(img_gray, kpmid)
        kplarge = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 1.0, _angle=kpmid.angle)
        kpsmall = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.25, _angle=kpmid.angle)
        kps = [kpsmall, kpmid, kplarge]
    elif n_kps==2:
        kpmid = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.65)
        compute_angle(img_gray, kpmid)
        kpsmall = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.30, _angle=kpmid.angle)
        kps = [kpsmall, kpmid]
    else:
        kpmid = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.65)
        compute_angle(img_gray, kpmid)
        kp = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 1.0, _angle=kpmid.angle)
        kps = [kp]
    return kps

def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1./(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)

def packSIFTOctave(octave, layer):
    if octave == -1:
        third_byte = int(255)
    else:
        third_byte = int(octave)
    second_byte = int(layer) << 8
    return second_byte | third_byte

def compute_octave(kps, layers = (-1,0,1,2,3,4), scale=1.0):
    b1 = 20 * scale; b2 = 40*scale; b3 = 60*scale; b4 = 80*scale; b5=100*scale
    for kp in kps:
        if kp.size <= b1: kp.octave = packSIFTOctave(layers[0],1)
        elif b1<kp.size<= b2: kp.octave = packSIFTOctave(layers[1],1)
        elif b2 < kp.size <= b3: kp.octave = packSIFTOctave(layers[2],1)
        elif b3 < kp.size <= b4: kp.octave = packSIFTOctave(layers[3],1)
        elif b4 < kp.size <= b5: kp.octave = packSIFTOctave(layers[4],1)
        else : kp.octave = packSIFTOctave(layers[5],1)

def get_channels(colorspace, indices):
    channels = []
    for i in indices:
        channels += [colorspace[i]]
    return channels

def compute_histogram(kps, channels, scale=1.0, std=sqrt(2)):
    n_channels = len(channels)
    for i in range(n_channels):
        channels[i] = ndimage.filters.gaussian_filter(channels[i], std)
    histograms = []
    l = len(kps)
    for i in range(l):
        kp = kps[i]
        x = int(kp.pt[0] * scale)
        y = int(kp.pt[1] * scale)
        kernel = int(kp.size * scale / 2.)
        hist = []
        for channel in channels:
            c_patch = channel[y - kernel:y + kernel, x - kernel:x + kernel]
            c_hist, _ = np.histogram(c_patch.flatten(), bins=50, range=(0, 255))
            hist += c_hist.tolist()
        histograms += [hist]
    return histograms

def compute_SIFT(kps, channels, sigma = 1.6):
    sift = cv2.xfeatures2d.SIFT_create(sigma=sigma)
    sifts = []
    l = len(kps)
    descriptors = []
    for c in channels:
        _, des = sift.compute(c, kps)
        descriptors += [des]
    for i in range(l):
        kp_descriptor = []
        for des in descriptors:
            kp_descriptor += des[i].tolist()
        sifts += [kp_descriptor]

    return sifts

def get_blob_descriptors(histograms, sift_des, n_kps):
    size = len(histograms)
    descriptors = []
    for i in range(0, size, n_kps):
        hdes = []
        sdes = []
        for j in range(n_kps):
            hdes += histograms[i+j]
            sdes += sift_des[i+j]
        descriptors += [hdes + sdes]
    return np.array(descriptors)

def load_model(path):
    model = pickle.load(open(path, "rb"), encoding='latin1')
    return model
