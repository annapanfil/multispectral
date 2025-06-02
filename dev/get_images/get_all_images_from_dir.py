import requests
import os

set_nr = 1

if set_nr == -1:
    response = requests.get("http://192.168.10.254/files/")
    print(response.text)
    set_nr = response.json().get("directories")[-1][0:4]

output_dir = f"/home/anna/Obrazy/multispectral/from_http/{set_nr:04}/"

if not os.path.exists(output_dir):
    print(f"creating directory {output_dir}")
    os.makedirs(output_dir)


url = f"http://192.168.10.254/files/{set_nr:04}SET/000/"

status_code = 200
image_nr = 0
channel_nr = 1

while status_code == 200:
    img_fn = f"IMG_{image_nr:04}_{channel_nr}.tif"
    photo_url = f"{url}{img_fn}"
    print(photo_url)
    response = requests.get(photo_url)

    status_code = response.status_code
    if status_code != 200:
        print(response.json().get("message"))

    output_file = os.path.join(output_dir, img_fn)
    print(f"Saving {img_fn}")

    with open(output_file, 'wb') as file:
        file.write(response.content)

    channel_nr += 1
    if channel_nr == 7: 
        channel_nr = 1
        image_nr += 1