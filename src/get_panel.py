"""Get one multispectral photo from the camera and save it to a directory."""

import os
import requests
from src.processing.consts import DATASET_BASE_PATH

if __name__ == "__main__":
    panel_dir = f"{DATASET_BASE_PATH}/raw_images/temp_panel" # where to save it

    if not os.path.exists(panel_dir):
        print(f"creating directory {panel_dir}")
        os.makedirs(panel_dir)

    url = "http://192.168.1.83/capture"
    params = {
        'block': 'true',
        'cache_raw': 31,   # All 5 bands (binary 11111)
        'store_capture': 'true', # Save to SD card
        'preview': 'false',
        'cache_jpeg': 0,
    }

    response = requests.get(url, params=params)
    print("captured image")
    print (response.json())

    response = response.json()
    # capture_nr = response.get("raw_storage_path").get("1").split("/")[2] # for now ommiting 000 directory from '/files/0010SET/000/IMG_0001_5.tif'

    for ch, path in response.get("raw_cache_path").items():
        # photo_nr = response.get("raw_storage_path").get(ch).split("/")[4]

        photo_url = "http://192.168.1.83" + path
        response_img = requests.get(photo_url)
        print("got image")
        

        output_file = os.path.join(panel_dir, f'IMG_0000_{ch}.tif')
        print(f"Saving channel {ch} to {output_file}")

        with open(output_file, 'wb') as file:
            file.write(response_img.content)
    
    print("panel photo saved")


