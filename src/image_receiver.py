import socket
import struct
import time
import zstandard as zstd
import os

# Zstandard decompressor
dctx = zstd.ZstdDecompressor()

output_dir = "../out/received"
os.makedirs(output_dir, exist_ok=True)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind(('0.0.0.0', 5000))
    server.listen(1)
    print("Waiting for connection...")
    conn, addr = server.accept()
    print(f"Connected by {addr}")
    print("Connection established " + str(time.time()))
    times = []
    with conn:
        count = 0
        while True:
            # receive the size of the filename and the filename
            filename_size_data = conn.recv(4)
            start = time.time()
            if not filename_size_data:
                break
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
                    break
                compressed += chunk
            raw = dctx.decompress(compressed)
            
            # save the image with the received filename
            with open(f"{output_dir}/{filename}", 'wb') as f:
                f.write(raw)
            print(f"Received compressed data of size: {len(compressed)}. Decompressed and saved to {output_dir}/{filename}")
            count += 1
            times.append(time.time() - start)
            print(f"Time taken to receive and save: {times[-1]:.2f} seconds")

        print(f"Total time taken: {sum(times):.2f} seconds  for {count} files (Now it's {time.time()})")
        print(f"Avg time per photo: {sum(times)/len(times):.2f}, max: {max(times):.2f}s, min: {min(times):.2f}s, median: {sorted(times)[len(times)//2]:.2f}s")
