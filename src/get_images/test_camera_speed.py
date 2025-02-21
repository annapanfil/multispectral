import os
from numpy import mean, std
import requests
from time import perf_counter


output_dir = "out"

if not os.path.exists(output_dir):
    print(f"creating directory {output_dir}")
    os.makedirs(output_dir)

url = "http://192.168.1.83/capture"
params = {
    "block": "true"
}

times = []
start = perf_counter()

for i in range(100):
    response = requests.get(url, params=params)

    response_json = response.json()

    capture_nr = response_json.get("raw_storage_path").get("1").split("/")[2] # for now ommiting 000 directory from '/files/0010SET/000/IMG_0001_5.tif'


    for ch, path in response_json.get("raw_cache_path").items():
        photo_nr = response_json.get("raw_storage_path").get(ch).split("/")[4]

        photo_url = "http://192.168.1.83" + path
        response = requests.get(photo_url)
        
        output_file = os.path.join(output_dir, photo_nr)

        with open(output_file, 'wb') as file:
            file.write(response.content)

    times.append(perf_counter()-start)
    print(f"downloaded photo {i}")
    start = perf_counter()



print(min(times), max(times), mean(times), std(times))