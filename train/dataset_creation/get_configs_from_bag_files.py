"""Create a config file for each altitude based on the images in a rosbag file.

Remember to init ros before executing this script.
"""

import pprint
import numpy as np
import yaml
import rosbag # you have to init ros

from src.processing.consts import DATASET_BASE_PATH

if __name__ == "__main__":
    # Configurations
    # bag_file = "matriceBag_multispectral_2016-02-11-18-02-40.bag"
    # experiment_name =  "mandrac2025_04_16_weights"
    # panel_image_nr = "0000"
    # images_with_litter = list(range(21, 230))

    # bag_file = "matriceBag_multispectral_2016-02-11-18-47-02.bag"
    # experiment_name = "mandrac2025_04_16_pile"
    # panel_image_nr = "0436"
    # images_with_litter = list(range(260,262)) + list(range(266,434))

    # bag_file = "matriceBag_multispectral_2016-02-11-18-59-05.bag"
    # experiment_name = "mandrac2025_04_16_transparent"
    # panel_image_nr = "0436"
    # images_with_litter = list(range(449, 600))

    # bag_file = "matriceBag_multispectral_2016-02-11-19-09-56.bag"
    # experiment_name = "mandrac2025_04_16_green"
    # panel_image_nr = "0436"
    # images_with_litter = list(range(602,665)) + list(range(667,826))

    # bag_file = "matriceBag_multispectral_2016-02-11-19-25-55.bag"
    # experiment_name = "mandrac2025_04_16_transparent_and_green"
    # panel_image_nr = "0436"
    # images_with_litter = list(range(839,900)) + list(range(907, 1095))

    # bag_file = "matriceBag_multispectral_2025-05-19-10-51-49.bag"
    # experiment_name = "hamburg_mapping"
    # panel_image_nr = "0000"
    # images_with_litter = list(range(106,483))

    bag_file = "matriceBag_multispectral_2025-05-15-12-04-05.bag"
    experiment_name = "hamburg_pile"
    panel_image_nr = "0017"
    images_with_litter = list(range(41, 298))



    files_path = f"{DATASET_BASE_PATH}/annotated/" + experiment_name + "/"
    raw_images_path =f"{DATASET_BASE_PATH}/raw_images/hamburg_2025_05_15/"
    config_files_path = "conf/processing"
    channels_out_path = f"{DATASET_BASE_PATH}/for_annotation/{experiment_name}/"
    topic_name = "/camera/trigger"
    tolerance = 2 # Tolerancja dla point.z
    heights = [5, 10, 15, 20, 25, 30] # Lista wysokości do sprawdzenia
    ##################
    frame_ids = {height: [] for height in heights}
    actual_heights = {height: [] for height in heights}

    # get ids of the frames on certain heights
    with rosbag.Bag(files_path + "/bags/" + bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            for height in heights:
                if abs(msg.point.z - height) <= tolerance: 
                    frame_ids[height].append(msg.header.frame_id) 
                    actual_heights[height].append(msg.point.z)
                    break


    # Filter out images with litter
    print("filtering out images with litter")
    image_numbers = {h: [int(path.split("/")[4].split("_")[1]) for path in frame_ids[h]] for h in heights} # get image numbers from paths
    for h in heights:
        for i in range(len(image_numbers[h])-1, -1, -1):
            if image_numbers[h][i] not in images_with_litter:
                del image_numbers[h][i]
                del actual_heights[h][i]
                del frame_ids[h][i]


    # save actual heights for the images
    with open(f'out/{experiment_name}_heights.txt', 'w') as f:
        f.write("\n".join([f"{h}m: {len(ids)} photos with litter" for h, ids in image_numbers.items()]) + '\n')

        for height in heights:
            im_n = image_numbers[height]
            rounded_heights = [round(h, 2) for h in actual_heights[height]]

            f.write(f"{height} m:\n")
            f.write(pprint.pformat(list(zip(im_n, rounded_heights))) + '\n')

    print(f"Zapisano wysokości do pliku out/{experiment_name}_heights.txt")

    def create_config(image_paths, height, suffix=""):
        paths = {
            "images": f"{raw_images_path}/images/", ## + set/subset",
            "warp_matrices": f"{DATASET_BASE_PATH}/annotated/warp_matrices",
            "panel_image_nr": panel_image_nr,
            "output": f"{raw_images_path}/chosen_images/", # + name/altitude
            "channels_output": channels_out_path # + name/altitude
        }

        paths["images"] += image_paths[0][7:19]
        paths["output"] += f"{height}/"
        paths["channels_output"] += f"{height}/"

        image_numbers = [int(path.split("/")[4].split("_")[1]) for path in image_paths]

        config = {
            "paths": paths,
            "params": {
                "altitude": height,
                "image_numbers": image_numbers
            }
        }
        config_filename = f"{experiment_name}_{height}m{suffix}.yaml"
        with open(f"{config_files_path}/{config_filename}", 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        return config_filename


    # Create yaml configuration for each height
    configs = []
    for height in heights:  
        if len(frame_ids[height]) == 0:
            print(f"No images for height {height}m")
            continue

        if frame_ids[height][0][7:19] != frame_ids[height][-1][7:19]: # subsets
            subsets = np.array([frame[7:19] for frame in frame_ids[height]])
            unique_subsets = np.unique(subsets)
            print(f"The images are in different subsets. Creating separate configs for subsets {unique_subsets}")
            frame_ids[height] = np.array(frame_ids[height])
            for subset in unique_subsets:
                config_fn = create_config(frame_ids[height][subsets == subset], height, suffix="_" + subset.split("/")[1])
                configs.append(config_fn.split(".")[0])
        else:
            config_fn = create_config(frame_ids[height], height)
            configs.append(config_fn.split(".")[0])

    print(f"Zapisano pliki konfiguracyjne do {','.join(configs)}")
