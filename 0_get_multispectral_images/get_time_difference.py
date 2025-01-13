import os
from exiftool import ExifToolHelper
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

photo_dir = '/home/anna/Obrazy/mandrac_2024_10_11/0020SET/000'
creation_times = []

# get only photo from the first camera, without panel photos
filenames = os.listdir(photo_dir)
filenames = list(filter(lambda filename: filename.endswith("_1.tif"), filenames))
filenames.remove("IMG_0000_1.tif")
filenames.remove("IMG_0099_1.tif")

filepaths = [os.path.join(photo_dir, file_name) for file_name in filenames]

with ExifToolHelper() as et:
    files_tags = et.get_tags(filepaths, tags=['CreateDate'])

creation_times = [datetime.strptime(d['EXIF:CreateDate'], '%Y:%m:%d %H:%M:%S') for d in files_tags]
creation_times.sort()

time_diffs = [
    (creation_times[i+1] - creation_times[i]).total_seconds()
    for i in range(len(creation_times) - 1)
]

# show statistics
if time_diffs:
    avg_time_diff = sum(time_diffs) / len(time_diffs)
    print(f'Time difference between the photos:\nmin: {min(time_diffs):.4f} s\navg:{avg_time_diff:.4f} s\nmax: {max(time_diffs):.4f} s')
else:
    print('Not enough data')

counts, bins, _ = plt.hist(time_diffs)
print(counts, bins)
plt.title("time between images")
plt.show()


# Get filenames with 0 s difference
time_diffs = np.array(time_diffs)
creation_times = np.array(creation_times)
zero_times = creation_times[:-1][time_diffs == 0]

zero_times_str = [dt.strftime('%Y:%m:%d %H:%M:%S') for dt in zero_times]

matching_files = []
for dt_str in zero_times_str:
    for file_tag in files_tags:
        if dt_str == file_tag['EXIF:CreateDate']:
            matching_files.append(file_tag)

print("Pliki z dopasowanymi datami:")
for file in matching_files:
    print(file)