import cv2
from matplotlib import pyplot as plt

from detection import detect_blobs, find_pool
from display import draw_litter, show_image, show_images

# -----------------------------------------------------------------
dog_threshold = 0.01
size_max_threshold = 1/10 # as percentage of the photo width
im_path = "../input/0021_10_meanRE.png"
sigma = 16

# -----------------------------------------------------------------

image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
pool = find_pool(image)

cropped_image = image[pool.y_b:pool.y_t, pool.x_l:pool.x_r]
blob_contours, dog_image, mask = detect_blobs(cropped_image, sigma, dog_threshold)

# show
im_pool = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
cv2.rectangle(im_pool, (pool.x_l, pool.y_b), (pool.x_r, pool.y_t), (0,0,255), 5)

im_litter = draw_litter(cropped_image, blob_contours, size_max_threshold, circles=False, rectangles=False, contours=True)

show_images([im_pool, dog_image, mask, im_litter])


