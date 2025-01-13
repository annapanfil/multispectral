import cv2
import numpy as np

# Path to the video file
name = "LARIAT_r"
video_path = f'/home/anna/Pictures/christmas_project/{name}.MP4'
output_video_path = f"/home/anna/Pictures/christmas_project/processed/Lariat_with_landing.MP4"
begin = 7
end = 51

# photo = cv2.imread("/home/anna/Pictures/christmas_project/IMG_5453_r.JPG")
# photo = cv2.resize(photo, (1920, 1080))

def apply_gamma_correction(frame, gamma=2.2):
    # Normalize pixel values to [0, 1], apply gamma correction, then scale back to [0, 255]
    corrected = np.power(frame / 255.0, gamma) * 255
    return corrected.astype(np.uint8)

# Open the video file
cap = cv2.VideoCapture(video_path)

frames = []
fps = 25  # Number of frames per second to extract
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Add frame to the array at specified intervals
    if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
        frames.append(frame)
    
    frame_count += 1

cap.release()

print(f"Extracted {len(frames)} frames.")

# przekształć
text_frames = frames[begin*fps: end*fps]
new_frames = []
current_max_frame = np.zeros_like(text_frames[0])

for frame in text_frames:
    # frame = cv2.addWeighted(frame, 0.3, photo, 0.7, 0)  # Blend frames (adjust weights as needed)
    # frame = apply_gamma_correction(frame)  # Apply gamma correction
    current_max_frame = np.maximum(current_max_frame, frame)  # Aktualizacja maksymalnych wartości
    new_frames.append(current_max_frame)

print(f"Processed {len(new_frames)} frames.")

# zapisz
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, fps*4, (width, height))

for frame in frames[:begin*fps]:
    for _ in range(4):
        video.write(frame)

for frame in new_frames:
    video.write(frame)

for frame in frames[end*fps:]:
    f = np.maximum(current_max_frame, frame)
    for _ in range(4):
        video.write(f)

video.release()
print(f"Wideo zapisane jako {output_video_path}")





