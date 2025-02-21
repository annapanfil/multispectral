import os
base_dir = "/home/anna/Datasets/for_annotation/"

for d in os.listdir(base_dir):
    if not os.path.isdir(base_dir + d):
        continue
    print(d)
    fn = base_dir + d + "/original_images.txt"
    if not os.path.exists(fn):
        print("No original_images.txt in dir " + d)
        continue
    with open(fn) as f:
        lines = f.readlines()
    
    with open(f"{base_dir}/original_images.txt", "a") as f:
        for line in lines:
            f.write(f"{d},{line.replace(' ', '')}")