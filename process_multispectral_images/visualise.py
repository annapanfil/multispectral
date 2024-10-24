import cv2
import numpy as np
import matplotlib.pyplot as plt

import micasense.imageutils as imageutils


def show_image(image, title="Image", figsize=(30,23), cmap='gray'):
    """Display an image with a given title."""
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
    

def save_image(image, filename, bgr = False):
    """Save an image to a file."""
    if bgr: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = (image * 255).astype(np.uint8)  # Scale to 0-255 for saving
    cv2.imwrite(filename, image)
    print("Saved to " + filename)


def get_components_view(img_aligned, img_type, band_indices=(2,1,0)):
    """visualise aligned image
    @param img_type type of image (reflectance or radiance)
    @param band_indices channels to display (2,1,0) for RGB
    """

    # # Create an empty normalized stack for viewing
    # im_display = np.zeros((img_aligned.shape[0], img_aligned.shape[1], img_aligned.shape[2]), dtype=np.float32)

    if img_type == 'reflectance' and band_indices == [2,1,0]: #rgb
        # for reflectance images we maintain white-balance by applying the same display scaling to all bands
        im_min = np.percentile(img_aligned[:,:,0:2].flatten(),  0.1)  # modify with these percentilse to adjust contrast
        im_max = np.percentile(img_aligned[:,:,0:2].flatten(), 99.9)  # for many images, 0.5 and 99.5 are good values
        for i in band_indices:
            img_aligned[:,:,i] =  imageutils.normalize(img_aligned[:,:,i], im_min, im_max)



    elif img_type == 'radiance' or band_indices != [2, 1, 0]:
        # for radiance images we do an auto white balance since we don't know the input light spectrum by
        # stretching each display band histogram to it's own min and max
        for i in band_indices:
            img_aligned[:,:,i] =  imageutils.normalize(img_aligned[:,:,i])

    im_visualised = img_aligned[:,:,band_indices]

    return im_visualised

def get_ndvi(img):
    return (img[:,:,3] - img[:,:,2]) / (img[:,:,3] + img[:,:,2])



def plot_one_channel(im_aligned, channel_nr=5, figsize=(30,23), out_fn=None, show=True):

    im_channel = im_aligned[:, :, channel_nr]
    
    # normalise
    min_val = im_channel.min()
    max_val = im_channel.max()

    if max_val - min_val > 0:
        normalized_channel = (im_channel - min_val) / (max_val - min_val) * 255
    else:
        normalized_channel = np.zeros(im_channel.shape)  # If all values are the same

    if show: show_image(normalized_channel, f"channel {channel_nr}", figsize, cmap="inferno")
    if out_fn: save_image(normalized_channel, out_fn)

    return normalized_channel
        

def plot_all_channels(im_aligned, out_fn=None, show=True):
    plt.subplots(3, 2, figsize=(16, 18))

    for channel in range(im_aligned.shape[2]):
        normalized_channel = plot_one_channel(im_aligned, channel, out_fn=None, show=False)
        
        plt.subplot(3, 2, channel+1)
        plt.axis('off') 
        plt.imshow(normalized_channel, cmap="inferno")
        plt.title(f"channel {channel}")

    if out_fn:
        plt.savefig(out_fn)
        print("Saved to " + out_fn)

    if show:
        plt.show()


# def enhance_rgb(img, figsize=(30,23), out_fn=None, show=True):

#     # Create an enhanced version of the RGB render using an unsharp mask
#     gaussian_rgb = cv2.GaussianBlur(img, (9,9), 10.0)
#     gaussian_rgb[gaussian_rgb<0] = 0
#     gaussian_rgb[gaussian_rgb>1] = 1
#     unsharp_rgb = cv2.addWeighted(img, 1.5, gaussian_rgb, -0.5, 0)
#     unsharp_rgb[unsharp_rgb<0] = 0
#     unsharp_rgb[unsharp_rgb>1] = 1

#     # Apply a gamma correction to make the render appear closer to what our eyes would see
#     gamma = 1.4
#     gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
#     plt.figure(figsize=figsize)
#     plt.imshow(gamma_corr_rgb, aspect='equal')
#     plt.title("Corrected_rgb")
#     plt.axis('off')

#     if out_fn: 
#         # plt.savefig(out_fn)
#         gamma_corr_rgb = (gamma_corr_rgb * 255).astype(np.uint8)  # Scale to 0-255 for saving
#         cv2.imwrite(out_fn, cv2.cvtColor(gamma_corr_rgb, cv2.COLOR_RGB2BGR))
#         print("Saved to " + out_fn)

#     if show:
#         plt.show()
