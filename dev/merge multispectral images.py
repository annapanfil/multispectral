import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import json

def read_exiftool(image_path):
    """Read camera parameters using ExifTool."""
    command = ['exiftool', '-json', image_path]
    
    try:
        # Execute the command and get output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Parse the JSON output
        metadata = json.loads(result.stdout)
        return metadata[0]  # return the first image's metadata
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None

def load_channel(image_path):
    """Load a channel image as grayscale."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def create_rgb_image(red_channel, green_channel, blue_channel):
    """Combine channels into RGB image."""
    return cv2.merge([blue_channel, green_channel, red_channel])  # OpenCV uses BGR format

def display_image(image, title):
    """Display an image using matplotlib."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_images(images, titles):
    """Display images using matplotlib."""
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def create_camera_matrix(principal_point, focal_length):
    """Create the camera matrix using the principal point and focal length."""
    cx, cy = principal_point
    return np.array([[focal_length, 0, cx],
                     [0, focal_length, cy],
                     [0, 0, 1]], dtype=np.float32)

def compute_homography(rig_relatives, rig_translations):
    """Compute the homography matrix based on rig parameters."""
    # Convert rig relatives and translations to numpy arrays
    rr = np.array(rig_relatives, dtype=np.float32)
    rt = np.array(rig_translations, dtype=np.float32)

    # Create a simple transformation matrix (this is just an example; adjust as needed)
    homography = np.eye(3)
    homography[0, 2] = rt[0]  # Translation in X
    homography[1, 2] = rt[1]  # Translation in Y

    return homography

def undistort_image(image, camera_matrix, distortion_coeffs):
    """Undistort the image using the camera matrix and distortion coefficients."""
    return cv2.undistort(image, camera_matrix, distortion_coeffs)

def align_images(base_image, target_image, homography):
    """Align target image to the base image using the homography matrix."""
    aligned_image = cv2.warpPerspective(target_image, homography, (base_image.shape[1], base_image.shape[0]))
    return aligned_image



path = '/home/anna/Obrazy/multispectral/0001SET/000/IMG_0000'

red_channel_path = path + "_1.tif"
green_channel_path = path + "_2.tif"
blue_channel_path = path + "_3.tif"

# Load channel images
red_channel = load_channel(red_channel_path)
green_channel = load_channel(green_channel_path)
blue_channel = load_channel(blue_channel_path)


# Read EXIF parameters from the red channel image
exif_params = read_exiftool(red_channel_path)
if exif_params:
    print("EXIF Parameters:")
    print("Principal Point: ", exif_params.get('PrincipalPoint', 'N/A'))
    print("Rig relatives: ", exif_params.get('RigRelatives', 'N/A'))
    print("Rig translations: ", exif_params.get('RigTranslations', 'N/A'))
    print("Focal length: ", exif_params.get('FocalLength', 'N/A'))
    print("Perspective distortion: ", exif_params.get('PerspectiveDistortion', 'N/A'))

principal_point = exif_params.get('PrincipalPoint', 'N/A') # x,y 
rig_relatives = exif_params.get('RigRelatives', 'N/A') 
rig_translations = exif_params.get('RigTranslations', 'N/A') # x, y, z
focal_length = exif_params.get('FocalLength', 'N/A')
distortion_coeffs = exif_params.get('PerspectiveDistortion', 'N/A')  # k1, k2, p1, p2, k3

camera_matrix = create_camera_matrix(principal_point, focal_length)

# Load the images (replace with your actual image paths)
# base_image = cv2.imread('path_to_red_channel.tif')
# target_image = cv2.imread('path_to_green_channel.tif')  # Replace with the actual channel image

# Undistort the images
red_undistorted = undistort_image(red_channel, camera_matrix, distortion_coeffs)
# target_image_undistorted = undistort_image(target_image, camera_matrix, distortion_coeffs)

# Compute the homography matrix
homography = compute_homography(rig_relatives, rig_translations)

# Align the target image to the base image
aligned_image = align_images(base_image_undistorted, target_image_undistorted, homography)

# Display the images
display_images([base_image_undistorted, target_image_undistorted, aligned_image], 
               ['Base Image Undistorted', 'Target Image Undistorted', 'Aligned Image'])



# print("EXIF Parameters:", exif_params)


# Create RGB image
# rgb_image = create_rgb_image(red_channel, green_channel, blue_channel)

# Display the RGB image
# display_image(rgb_image, 'Combined RGB Image')


# --------------------------------------
# import numpy as np
# from PIL import Image

# def normalize(channel_array):
#     max_val = channel_array.max()
#     return (channel_array/max_val *255).astype(np.uint8)


# path = '/home/anna/Obrazy/multispectral/0001SET/000/IMG_0000'
# # Load the channels separately (assuming they're stored as grayscale images)
# red_channel = np.array(Image.open(path + '_1.tif'))
# green_channel = np.array(Image.open(path + '_2.tif'))
# blue_channel = np.array(Image.open(path + '_3.tif'))


# red_channel = normalize(red_channel)
# green_channel = normalize(green_channel)
# blue_channel = normalize(blue_channel)

# # red_channel_image = Image.fromarray(red_channel)
# # red_channel_image.show()

# # Stack the channels to create an RGB image
# rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

# # Convert back to an image using Pillow
# rgb_image_pil = Image.fromarray(rgb_image)

# # Save or display the image
# rgb_image_pil.show()
# # rgb_image_pil.save('~/Obrazy/multispectral/output_image.png')