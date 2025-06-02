from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import subprocess

# https://www.dji.com/hr/downloads/softwares/dji-thermal-sdk

lib_path = "/home/anna/Obrazy/dji_thermal_sdk/utility/bin/linux/release_x64"
source =  "/home/anna/Obrazy/DJI_20241008095751_0004_T.JPG"

output = os.path.splitext(source)[0] + ".raw"


os.environ['LD_LIBRARY_PATH'] = lib_path
res = subprocess.run([lib_path + "/dji_irp", "-s", source, "-a", "measure", "-o", output],
                capture_output=True)

if res.returncode != 0:
    print(res.stdout)

else:
    temps = np.fromfile(output, dtype=np.int16).reshape((512, 640))
    print(temps.min()/10, temps.max()/10)

    np.savetxt( os.path.splitext(source)[0] + "_temps.csv", temps/10, delimiter=',', fmt="%.1f")

    plt.imshow(temps.astype(np.uint), cmap="inferno")
    plt.show()
