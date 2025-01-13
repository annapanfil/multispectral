import cv2
import os

image_folder = "/home/anna/Datasets/video/0044SET/000"
video_name = 'out/video_3_hz_non_blocking.avi'

images = sorted([img for img in os.listdir(image_folder) if img.endswith("_1.tif")])
# images = [f"IMG_{i:04d}_1.tif" for i in range(140, 217)]

print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 3, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()


#3-9
#10-20
#21-32
