import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import rosbag
from ultralytics import YOLO

from detection.shapes import Rectangle
from detection.utils import get_real_piles_size, greedy_grouping, prepare_image, time_decorator
from processing.load import align_from_saved_matrices, get_irradiance, load_image_set

def main():
    """Main function for detection and pile positioning."""
    # POOL_HEIGHT = 0.5  # m
    # CAM_HFOV = np.deg2rad(49.6)  # rad
    # CAM_VFOV = np.deg2rad(38.3)  # rad

    img_dir = "/home/anna/Datasets/raw_images/mandrac_2024_12_6/images/" 
    bag_path = "/home/anna/Datasets/annotated/mandrac/bag_files/matriceBag_multispectral_2024-12-06-09-49-22.bag"
    model_path = "../models/sea-form8_sea_aug-random_best.pt"
    warp_matrices_dir = "/home/anna/Datasets/annotated/warp_matrices"
    topic_name = "/camera/trigger"
    out_path = "/home/anna/Datasets/annotated/mandrac/detections_new_model.mp4"
    fps = 3 #25
    panel_img_nr = "0004"
    start_from = 21
    end_on = 204

    new_image_size = (800, 608)
    formula = "(N - (E - N))"
    channels = ["N", "G", formula] # wrong order, but thats how the model was trained #TODO: fix this
    is_complex = True if any(len(form) for form in channels) > 40 else False
        
    start = time.time()
    # get images and altitudes from bagfiles
    pred_images = []
    model = YOLO(model_path)
    
    with rosbag.Bag(bag_path, 'r') as bag:
        try:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                img_start = time.time()
                altitude = round(msg.point.z)
                set_nr = "/".join(msg.header.frame_id.split("/")[2:4])
                img_nr = msg.header.frame_id.split("/")[-1].split("_")[1]
                
                if int(img_nr) < start_from: 
                    continue
                if int(img_nr) > end_on: break

                print(f"Processing image {img_nr} with altitude {altitude}")
                
                # Read the channels, align and convert to desired format
                img_capt, panel_capt = load_image_set(img_dir + set_nr, img_nr, panel_img_nr, no_panchromatic=True)
                img_type = get_irradiance(img_capt, panel_capt, display=False, vignetting=False)
                img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices_dir, altitude, allow_closest=True, reference_band=0)
                
                image = prepare_image(img_aligned, channels, is_complex, new_image_size)
            
                # Predict
                results = model.predict(source=image, conf=0.5, save=False)
                pred_bbs = results[0].boxes.xyxy.cpu().numpy()
                pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

                # Merge piles
                merged_bbs, merged_img = greedy_grouping(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=True)

                # TODO: Get position in the world
                # sizes = get_real_piles_size(image.shape[:2], altitude - POOL_HEIGHT, CAM_HFOV, CAM_VFOV, merged_bbs)
            
                # for rect, size in zip(merged_bbs, sizes):
                #     rect.draw(image, color=(0, 255, 0), thickness=2)
                    # text = f"{size[0]*100:.0f}x{size[1]*100:.0f}" # cm
                    # cv2.putText(image, text, (rect.x_l, rect.y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                for rect in merged_bbs:
                    rect.draw(image, color=(0, 255, 0), thickness=2)

                pred_images.append(image)
                print("Processing time " + str(time.time()-img_start)
        except FileNotFoundError as e:
            print(e)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, new_image_size)

    for image in pred_images:
        video.write(image)

    # cv2.destroyAllWindows()
    video.release()

    print("----\nWhole main took", time.time() - start, "s")

if __name__ == "__main__":
    main()
