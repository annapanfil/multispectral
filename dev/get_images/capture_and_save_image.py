"""Capture one photo using Micasense RedEdge-P http request and save all channels to separate files on disk."""

import requests
import os

if __name__ == "__main__":
    output_dir = "/home/anna/Obrazy/multispectral/from_http/"

    url = "http://192.168.10.254/capture"
    params = {
        "block": "true",
        "cache_jpg": "31"
    }

    response = requests.get(url, params=params)
    print("captured image")
    print (response.json())

    response = response.json()
    capture_nr = response.get("raw_storage_path").get("1").split("/")[2] # for now ommiting 000 directory from '/files/0010SET/000/IMG_0001_5.tif'

    if not os.path.exists(output_dir):
        print(f"creating directory {output_dir}")
        os.makedirs(output_dir)

    for ch, path in response.get("raw_cache_path").items():
        photo_nr = response.get("raw_storage_path").get(ch).split("/")[4]

        photo_url = "http://192.168.10.254" + path
        response = requests.get(photo_url)
        print("got image")
        

        output_file = os.path.join(output_dir, capture_nr, photo_nr)
        print(f"Saving channel {ch} to {output_file}")

        with open(output_file, 'wb') as file:
            file.write(response.content)

