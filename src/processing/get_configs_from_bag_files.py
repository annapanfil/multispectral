import pprint
import numpy as np
import yaml
import rosbag # you have to init ros

bag_file = "matriceBag_multispectral_2024-12-06-10-33-18.bag"
experiment_name = "transparent_sea"
panel_image_nr = "0000"

# bag_file = "matriceBag_multispectral_2024-12-06-11-09-36.bag"
# experiment_name = "green_sea"
# panel_image_nr = "0000"# taken from green_marina

# bag_file = "matriceBag_multispectral_2024-12-06-11-35-53.bag"
# experiment_name = "green_marina"
# panel_image_nr = "0000"

# bag_file = "matriceBag_multispectral_2024-12-06-11-56-00.bag"
# experiment_name = "transparent_marina"
# panel_image_nr = "0000" 


bag_files_path = "/home/anna/Datasets/mandrac_2024_12_6/bags"
config_files_path = "conf/processing"
topic_name = "/camera/trigger"
tolerance = 2  # Tolerancja dla point.z
heights = [10]  # Lista wysokości do sprawdzenia #[10, 15, 20, 25, 30]

frame_ids = {height: [] for height in heights}
actual_heights = {height: [] for height in heights}

# get ids of the frames on certain heights
with rosbag.Bag(bag_files_path + "/" + bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        for height in heights:
            if abs(msg.point.z - height) <= tolerance: 
                frame_ids[height].append(msg.header.frame_id) 
                actual_heights[height].append(msg.point.z)
                break

# save actual heights for the images
with open(f'./out/{experiment_name}_heights.txt', 'w') as f:
    f.write("\n".join([f"{h}m: {len(ids)} photos" for h, ids in frame_ids.items()]) + '\n')

    for height in heights:
        image_numbers = [int(path.split("/")[4].split("_")[1]) for path in frame_ids[height]]
        rounded_heights = [round(h, 2) for h in actual_heights[height]]

        f.write(f"{height} m:\n")
        f.write(pprint.pformat(list(zip(image_numbers, rounded_heights))) + '\n')



# def create_config(image_paths, height, suffix=""):
#     paths = {
#         "images": "/home/anna/Datasets/mandrac_2024_12_6/images/", ## 0041SET/000",
#         "warp_matrices": "/home/anna/Datasets/annotated/warp_matrices",
#         "panel_image_nr": panel_image_nr,
#         "output": "/home/anna/Datasets/mandrac_2024_12_6/chosen_images/", #altitude
#         "channels_output": "/home/anna/Datasets/for_annotation/mandrac/" # altitude
#     }

#     paths["images"] += image_paths[0][7:19] # add set and subset
#     paths["output"] += f"{experiment_name}/{height}/"
#     paths["channels_output"] += f"{experiment_name}/{height}/"

#     image_numbers = [int(path.split("/")[4].split("_")[1]) for path in image_paths]

#     config = {
#         "paths": paths,
#         "params": {
#             "altitude": height,
#             "image_numbers": image_numbers
#         }
#     }
#     config_filename = f"{experiment_name}_{height}m{suffix}.yaml"
#     with open(f"{config_files_path}/{config_filename}", 'w') as file:
#         yaml.dump(config, file, default_flow_style=False)

#     print(f"Plik konfiguracyjny dla wysokości {height} zapisany jako {config_filename}")


# Create yaml configuration for each height
# for height in heights:  
#     if frame_ids[height][0][7:19] != frame_ids[height][-1][7:19]:
#         subsets = np.array([frame[7:19] for frame in frame_ids[height]])
#         unique_subsets = np.unique(subsets)
#         print(f"The images are in different subsets. Creating separate configs for subsets {unique_subsets}")
#         frame_ids[height] = np.array(frame_ids[height])
#         for subset in unique_subsets:
#             create_config(frame_ids[height][subsets == subset], height, suffix="_" + subset.split("/")[1])
#     else:
#         create_config(frame_ids[height], height)


# Create yaml configuration for each height
for height in heights:  
    paths = {
        "images": "/home/anna/Datasets/mandrac_2024_12_6/images/", ## + xxxxSET,
        "warp_matrices": "/home/anna/Datasets/annotated/warp_matrices",
        "panel_image_nr": panel_image_nr,
        "output": "/home/anna/Datasets/mandrac_2024_12_6/chosen_images/", # + altitude
        "channels_output": "/home/anna/Datasets/for_annotation/mandrac/" # + altitude
    }

    paths["images"] += frame_ids[height][0].split("/")[2] # add set,

    paths["output"] += f"{experiment_name}/{height}/"
    paths["channels_output"] += f"{experiment_name}/{height}/"

    image_numbers = [int(path.split("/")[4].split("_")[1]) for path in frame_ids[height]]

    config = {
        "paths": paths,
        "params": {
            "altitude": height,
            "image_numbers": image_numbers
        }
    }
    config_filename = f"{experiment_name}_{height}m.yaml"
    with open(f"{config_files_path}/{config_filename}", 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Plik konfiguracyjny dla wysokości {height} zapisany jako {config_filename}")





