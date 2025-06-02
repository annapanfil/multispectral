import os
import socket
import struct
import zstandard as zstd
import time

# Zstandard compressor
cctx = zstd.ZstdCompressor(level=3, threads=8)

image_dir = os.path.expanduser("~/Pictures/multispectral")
image_paths = sorted(os.listdir(image_dir))

print("start" + str(time.time()))
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect(('10.2.119.163', 5000))
    for filename in image_paths:
        full_path = os.path.join(image_dir, filename)
        with open(full_path, 'rb') as f:
            raw = f.read()
        compressed = cctx.compress(raw)
        filename_bytes = filename.encode("utf-8")
        sock.sendall(struct.pack('!I', len(filename_bytes)) + filename_bytes + struct.pack('!I', len(compressed)) + compressed)