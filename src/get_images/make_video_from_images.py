import cv2
import os

image_folder = "/home/anna/Datasets/mandrac_2025_04_04/chosen_images/video/beach/"
video_name = '/home/anna/Datasets/mandrac_2025_04_04/chosen_images/video/beach.mp4'

images = sorted([img for img in os.listdir(image_folder)])
# images = sorted([img for img in os.listdir(image_folder) if img.endswith("_1.tif")])
# images = [f"IMG_{i:04d}_1.tif" for i in range(140, 217)]

print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 25 # 3
video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

