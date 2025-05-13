import gc
import os
import time
import click
import cv2
import numpy as np
import rosbag
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import queue
import threading
from detection.shapes import Rectangle
from detection.utils import greedy_grouping, prepare_image
from processing.load import align_from_saved_matrices, get_irradiance, load_all_warp_matrices, load_image_set

def main_processing(img_dir, bag_path, out_path, model_path, panel_img_nr, start_from, end_on):
    # Configuration (keep your original parameters)
    warp_matrices_dir = "/home/anna/Datasets/annotated/warp_matrices"
    topic_name = "/camera/trigger"
    fps = 9   # About 3 times faster than image acquisition
    new_image_size = (800, 608)
    formula = "(N - (E - N))"
    channels = ["N", "G", formula]
    is_complex = len(formula) > 40
    batch_size = 4 
    num_workers = 6 

    # Pre-load all warp matrices into memory
    warp_matrices = load_all_warp_matrices(warp_matrices_dir)

    # Initialize model with optimizations
    model = YOLO(model_path)
    model.fuse() # Fuse Conv+BN layers
    model.half()
    model.conf = 0.5

    # Parallel processing pipeline
    output_queue = queue.PriorityQueue()

    # Video writer thread
    video_thread = threading.Thread(target=video_writer, args=(output_queue, out_path, new_image_size, fps, start_from))
    video_thread.start()

    # Process messages in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Load all messages
        messages = []
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                img_nr = int(msg.header.frame_id.split("/")[-1].split("_")[1])
                if start_from <= img_nr <= end_on:
                    messages.append(msg)

        total_images = len(messages)
        start_time = time.time()

        # Process images in parallel batches
        future_map = {}
        batch = []
        for i, msg in enumerate(messages):
            future = executor.submit(process_image, msg, img_dir, panel_img_nr, 
                                    warp_matrices, channels, new_image_size, is_complex)
            future_map[future] = i
            batch.append(future)

            print_progress(i, total_images, start_time)
            
            if len(batch) >= batch_size:
                process_batch(batch, model, output_queue, future_map)
                batch = []

        # Process remaining images
        if batch:
            process_batch(batch, model, output_queue, future_map)

    # Finalize video
    output_queue.put((float('inf'), None))
    video_thread.join()

    return total_images


def print_progress(current, total, start_time):
    elapsed = time.time() - start_time
    percent = (current / total) * 100
    remaining = (elapsed / current) * (total - current) if current > 0 else 0
    
    print(f"\rProgress: {current}/{total} ({percent:.1f}%) | "
          f"Elapsed: {elapsed:.1f}s | "
          f"ETA: {remaining:.1f}s", end="", flush=True)


def process_image(msg, img_dir, panel_img_nr, warp_matrices, channels, new_image_size, is_complex):
    # Extract metadata
    altitude = round(msg.point.z)
    set_nr = "/".join(msg.header.frame_id.split("/")[2:4])
    img_nr = msg.header.frame_id.split("/")[-1].split("_")[1]

    # Load and process image
    img_capt, panel_capt = load_image_set(img_dir + set_nr, img_nr, panel_img_nr, no_panchromatic=True)
    img_type = get_irradiance(img_capt, panel_capt, display=False, vignetting=False)
    img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices, altitude, allow_closest=True, reference_band=0)
    image = prepare_image(img_aligned, channels, is_complex, new_image_size)
    
    return int(img_nr), image


def process_batch(batch, model, output_queue, future_map):
    # Get results in completion order
    results = []
    for future in batch:
        img_nr, image = future.result()
        results.append((img_nr, image))
        del future
        gc.collect()
    
    # Batch prediction
    images = [img for _, img in sorted(results, key=lambda x: x[0])]
    batch_results = model(images, augment=False, verbose=False)

    # Post-process and enqueue
    for (img_nr, _), results in zip(sorted(results, key=lambda x: x[0]), batch_results):
        merged_bbs, _ = greedy_grouping(
            [Rectangle(*bb.xyxy[0].cpu().numpy(), "rect") for bb in results.boxes],
            images[0].shape[:2],
            resize_factor=1.5
        )
        
        # Draw on image
        image = images.pop(0)
        for rect in merged_bbs:
            rect.draw(image, color=(0, 255, 0), thickness=2)
        
        output_queue.put((img_nr, image))

        # Clear processed data
        del merged_bbs, results
        gc.collect()

    # Clear batch memory
    del batch, images, batch_results
    gc.collect()

def video_writer(output_queue, out_path, size, fps, start_from):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, size)
    
    next_frame = start_from
    buffer = {}

    while True:
        frame_idx, frame = output_queue.get()
        if frame is None:
            break
        
        buffer[frame_idx] = frame
        
        while next_frame in buffer:
            video.write(buffer.pop(next_frame))
            next_frame += 1
    
    video.release()

@click.command()
@click.option('--img_dir', '-i', default="mandrac_2024_12_6", help='Directory with images (name in raw_images dir)')
@click.option('--bag_path', '-b', default="mandrac/bag_files/matriceBag_multispectral_2024-12-06-09-49-22.bag", help='Path to the bag file (relative to annotated dir)')
@click.option('--out_path', '-o', help='Output video path (relative to predicted_videos dir)')
@click.option('--model_path', '-m', default="../models/sea-form8_sea_aug-random_best.pt", help='Path to the YOLO model')
@click.option('--panel', '-p', default="0000", help='Panel image number (for alignment)')
@click.option('--start', '-s', default = 0, type=int, help='Start frame number')
@click.option('--end', '-e', default = 1000000, type=int, help='End frame number')
def main(img_dir, bag_path, out_path, model_path, panel, start, end):
    """
    Main function to process images and create a video with detections.
    """
    # Start the timer
    start_time = time.time()
    base_dir_raw = "/home/anna/Datasets/raw_images"
    base_dir_annotated = "/home/anna/Datasets/annotated/"
    base_dir_out = "/home/anna/Datasets/predicted_videos/"
    n_images = main_processing(f"{base_dir_raw}/{img_dir}/images/" , base_dir_annotated + bag_path, base_dir_out + out_path, model_path, panel, start, end)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds for {n_images} images")

if __name__ == "__main__":
    main()
    
    # dla 183 obrazków
    # 416s bez optymalizacji
    # 376s z wątkiem dla każdego obrazka
    # 230s z wątkiem dla każdego obrazka, wątkiem dla video, batch processingiem, pre-loading of warp matrices, fused and half YOLO
    # 246s bez half
