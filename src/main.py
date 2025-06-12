from pathlib import Path
import subprocess
import time
import click
# import rospy
# from geometry_msgs.msg import PointStamped
# from sensor_msgs.msg import Image
import exiftool
# from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
from src.processing.consts import DATASET_BASE_PATH
from src.main_drone import get_image_from_camera
# from src.main_ground import send_outcomes

import micasense.capture as capture
from src.processing.load import align_from_saved_matrices, find_images, get_irradiance, load_all_warp_matrices
from src.shapes import Rectangle
from src.utils import greedy_grouping, prepare_image
import gc

current_altitude = 15

def position_callback(msg):
    global current_altitude
    current_altitude = msg.point.z


@click.command()
@click.option("--debug", '-d', is_flag=True, default=False, help="Run in debug mode with local images")
def main(debug):
    """Main function to capture images from UAV multispectral camera and detect litter."""
    # rospy.init_node("UAV_multispectral_detector")
    # rospy.loginfo("Node has been started")
    print("Starting UAV multispectral litter detection...")

    ### CONFIGURATION
    formula = "(N - (E - N))"
    channels = ["N", "G", formula]
    is_complex = False
    new_image_size = (800, 608)
    original_image_size = (1456, 1088)

    warp_matrices_dir = f"{DATASET_BASE_PATH}/warp_matrices"
    # model_path = "models/sea-form8_sea_aug-random_best.pt"
    model_path = "models/model.onnx"
    panel_path = f"{DATASET_BASE_PATH}/raw_images/temp_panel" # here is saved the panel image when src.get_panel is used
    panel_nr = "0000"

    ### INITIALIZATION
    times = []
    # position_sub = rospy.Subscriber("/dji_osdk_ros/local_position", PointStamped, callback=position_callback)   

    warp_matrices = load_all_warp_matrices(warp_matrices_dir)
    panel_names = find_images(Path(panel_path), panel_nr, panel=True, no_panchromatic=True)
    panel_capt = capture.Capture.from_filelist(panel_names)
    
    # pos_pixel_pub = rospy.Publisher("/multispectral/pile_pixel_position", PointStamped, queue_size=10)
    # image_pub = rospy.Publisher("/multispectral/detection_image", Image, queue_size=10)

    print("Loading model...")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = session.get_inputs()[0].type
    # model = YOLO(model_path)
    # model.fuse() # Fuse Conv+BN layers
    # model.half()
    # model.conf = 0.5

    if not debug:
        url = "http://192.168.1.83/capture"
        params = {
            "block": "true",
            "cache_jpg": "31"
        }

        # TEST CAMERA CONNECTION
        print("Testing camera connection...")
        # rospy.loginfo("Testing camera connection...")
        try:
            output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            rospy.logerr("Failed to ping the camera")
            exit(1)
    else:
        # get image numbers from bag file
        # IMAGE_DIR = f"{DATASET_BASE_PATH}/raw_images/hamburg_2025_05_19/images/"
        # TOPIC_NAME = "/camera/trigger"
        pass

    ### MAIN LOOP
    # while True:
    for i in range(59, 74):
    # while not rospy.is_shutdown():
        start = time.time()

        # rospy.loginfo("Capturing photo...")
        print("Capturing photo...")
        altitude = current_altitude
        if not debug:
            # get images from camera
            paths = get_image_from_camera(url, params)
        else:
            storage_path = f"/home/lariat/images/img_to_simple_test/IMG_08{i}_1.tif"
            paths = [storage_path.replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]

            # get images from local directory
            # try:
            #     msg = rospy.wait_for_message(TOPIC_NAME, PointStamped, timeout=10)
            #     storage_path = msg.header.frame_id.replace("/files/", IMAGE_DIR)
            #     paths = [storage_path.replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]
            # except rospy.ROSException:
            #     rospy.logwarn(f"No message received on topic {TOPIC_NAME} within 10s. Retrying...")
            #     continue
            
        # group_key = paths[0].rsplit("/", 1)[1].rsplit("_", 1)[0] # for logging purposes

        # Preprocessing
        # print(f"Processing {group_key} with altitude {altitude:.0f} m")
        # rospy.loginfo(f"Processing {group_key} with altitude {altitude:.0f} m")
        
        img_capt = capture.Capture.from_filelist(paths)
        img_type = get_irradiance(img_capt, panel_capt, display=False, vignetting=False)
        img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices, altitude, allow_closest=True, reference_band=0)
        image = prepare_image(img_aligned, channels, is_complex, new_image_size)

        input = np.transpose(image, (2, 1, 0))
        # Add batch dimension: (1, C, H, W)
        input = np.expand_dims(input, axis=0)

        # Detection
        print("Detecting litter...")
        # results = model.predict(source=image, save=False, verbose=False)
        results = session.run(None, {input_name: input})

        bbs = []
        for pred_bbs in results[0]:
            for bb in pred_bbs:
                x1, y1, x2, y2, conf, cls = bb
                # filter if needed: e.g., if conf > 0.5
                if conf > 0.1:
                    # rect = Rectangle(x1, y1, x2, y2, "rect")
                    rect = Rectangle(y1, x1, y2, x2, "rect")
                    bbs.append(rect)
        
        pred_bbs = bbs
        # pred_bbs = results[0].boxes.xyxy.cpu().numpy()
        # pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

        # Merging
        print("Merging piles...")
        merged_bbs, _, _ = greedy_grouping(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=False)

        print(merged_bbs)
        image = image * 255.0  # Convert to uint8
        image = image.astype(np.uint8)
        for rect in merged_bbs:
            rect.draw(image, color=(0, 255, 0), thickness=2)
        
        import cv2
        cv2.imwrite(f"/home/lariat/images/preds/pred{i}.jpg", image)

        # print(f"Found {len(merged_bbs)} litter piles in {group_key}")
        # Sending outcomes
        # send_outcomes(merged_bbs, image, group_key, image_pub, pos_pixel_pub)

        end = time.time()
        print(f"Time for this photo: {end - start:.2f} seconds")
        # rospy.loginfo(f"Time for this photo: {end - start:.2f} seconds")
        times.append(end - start)
        
        del img_capt
        del img_aligned
        del image
        del pred_bbs
        del merged_bbs
        del results
        del input
        del bbs
        gc.collect()  # import gc

    
    if len(times) != 0:
        print("TIME STATISTICS:")
        print("--------------------")
        print("min: {}, max: {}, avg: {}".format(min(times), max(times), sum(times)/len(times)))
    else:
        print("No images captured, time statistics not available.")

if __name__ == "__main__":
    main()
