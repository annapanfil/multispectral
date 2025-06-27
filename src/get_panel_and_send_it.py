"""Get one multispectral photo from the camera and save it to a directory."""

from collections import defaultdict
import os
import socket
import struct
import time

import rospy
import zstandard as zstd
from src.processing.consts import DATASET_BASE_PATH

def receive_and_unpack_data(conn, decompressor):
    # receive the size of the filename and the filename
    try:
        filename_size_data = conn.recv(4)
    except socket.timeout:
        return None
    
    if not filename_size_data:
        return None
    (filename_size,) = struct.unpack('!I', filename_size_data)
    filename_b = conn.recv(filename_size)
    filename = filename_b.decode('utf-8')

    # receive the size of the compressed image and the image 
    size_data = conn.recv(4)
    (size,) = struct.unpack('!I', size_data)
    
    compressed = b''
    while len(compressed) < size:
        chunk = conn.recv(size - len(compressed))
        if not chunk:
            raise ConnectionError("Socket closed before receiving full compressed data")
        compressed += chunk
    raw = decompressor.decompress(compressed)

    return filename, raw


def handle_incoming_image(filename, image_data, image_groups, panel_dir):
    group_key = filename.rsplit("_", 1)[0]
    image_groups[group_key].append(image_data)
    print(f"Received image {filename} ({len(image_groups[group_key])} / 5)")

    if len(image_groups[group_key]) == 5:
        process_whole_img(image_groups[group_key], panel_dir)
        return True
    return False


def process_whole_img(images, panel_dir):
    for i, img_data in enumerate(images):
        path = os.path.join(panel_dir, f"IMG_0000_{i}.tiff")
        with open(path, "wb") as f:
            f.write(img_data)

    print("saved_panel")

def main():
    rospy.init_node("multispectral_panel_handler")
    rospy.loginfo("Node has been started")

    panel_dir = f"{DATASET_BASE_PATH}/raw_images/temp_panel" # where to save it

    image_groups = defaultdict(list)

    # Pre-load all warp matrices and panel
    global panel_capt 
    global warp_matrices
    global pos_pixel_pub
    global image_pub

    decompressor = zstd.ZstdDecompressor() 

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('0.0.0.0', 5000))
        server.listen(1)
        print("Waiting for connection...")
        server.settimeout(1)
        conn = None
        while not rospy.is_shutdown():
            try:
                conn, addr = server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except socket.timeout:
                continue
            break
        if conn is not None:
            print(f"Connected by {addr}")
            print("Connection established " + str(time.time()))
            
            with conn:
                conn.settimeout(20.0)
                while not rospy.is_shutdown():
                    res = receive_and_unpack_data(conn, decompressor)
                    if res is None:
                        continue
                    filename, raw = res
                    rospy.sleep(1)
                    
                    if handle_incoming_image(filename, raw, image_groups, panel_dir):
                        break

        print("Finishing")
        server.shutdown(socket.SHUT_RDWR)  
        server.close() 


if __name__ == '__main__':
    main()
