import cv2
import numpy as np
import matplotlib.pyplot as plt

import micasense.imageutils as imageutils
from src.processing.consts import CHANNELS, EPSILON

def show_image(image, title="Image", figsize=(30,23), cmap='gray'):
    """Display an image with a given title."""
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
    

def save_image(image, filename, gray = False, is_float=True):
    """Save an image to a file."""
    print(image.shape)
    if not gray: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if is_float: image = (image * 255).astype(np.uint8)  # Scale to 0-255 for saving
    cv2.imwrite(filename, image)
    print("Saved to " + filename)


def get_components_view(img_aligned, band_indices, gamma=2):
    """Visualise aligned image using the provided band indices and apply gamma correction."""
    img = np.zeros_like(img_aligned, dtype=np.float32)

    for i in range(0, img.shape[2]):
        img[:,:,i] =  imageutils.normalize(img_aligned[:,:,i])

    img = np.power(img, 1.0 / gamma)
    return img[:, :, band_indices]


def get_index_view(img_aligned, band1, band2):
    """Visualise image using two provided band indices. Used for NDVI, NDWI and other similar indices with a formula: (band1 - band2) / (band1 + band2)"""

    for i in [band1, band2]:
        img_aligned[:,:,i] =  imageutils.normalize(img_aligned[:,:,i])

    ndvi_image = (img_aligned[:, :, band1] - img_aligned[:, :, band2]) / (img_aligned[:, :, band1] + img_aligned[:, :, band2] + EPSILON)
    ndvi_image = (ndvi_image - np.min(ndvi_image)) / (np.max(ndvi_image) - np.min(ndvi_image))
    return ndvi_image


def get_SR_image(img_aligned):
    """Get normalised simple ratio (SR) image."""
    sr_image = img_aligned[:, :, CHANNELS["NIR"]] / (img_aligned[:, :, CHANNELS["R"]] + EPSILON)
    sr_image = (sr_image - np.min(sr_image)) / (np.max(sr_image) - np.min(sr_image))
    return sr_image

def get_PI_image(img_aligned):
    """Get normalised plastic index (PI) image."""
    pi_image = img_aligned[:, :, CHANNELS["NIR"]] / (img_aligned[:, :, CHANNELS["NIR"]] + img_aligned[:, :, CHANNELS["R"]] + EPSILON)
    pi_image = (pi_image - np.min(pi_image)) / (np.max(pi_image) - np.min(pi_image))
    return pi_image

def plot_one_channel(im_aligned, channel_nr=5):
    """Plot a single normalised channel from the aligned image."""

    im_channel = im_aligned[:, :, channel_nr]
    
    # normalise
    min_val = im_channel.min()
    max_val = im_channel.max()

    if max_val - min_val > 0:
        normalized_channel = (im_channel - min_val) / (max_val - min_val) * 255
    else:
        normalized_channel = np.zeros(im_channel.shape)  # If all values are the same

    return normalized_channel
        

def plot_all_channels(im_aligned, out_fn=None, show=True):
    """Plot all normalised channels from the aligned image in subplots."""
    plt.subplots(3, 2, figsize=(16, 18))

    for channel in range(im_aligned.shape[2]):
        normalized_channel = plot_one_channel(im_aligned, channel)
        
        plt.subplot(3, 2, channel+1)
        plt.axis('off') 
        plt.imshow(normalized_channel, cmap="gray")
        plt.title(f"channel {channel}")

    if out_fn:
        plt.savefig(out_fn)
        print("Saved to " + out_fn)

    if show:
        plt.show()
