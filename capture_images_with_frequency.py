import requests
from time import perf_counter
from loop_rate_limiters import RateLimiter
import numpy as np


url = "http://192.168.1.83/capture"
params = {
    "block": "false"
}

rate = RateLimiter(frequency=3)
times = []
start = perf_counter()
end = start + 60

print("Capturing images...")
try:
    i=0
    while perf_counter() < end:
        response = requests.get(url, params=params)
        # print(response.json())
        print("Captured photo", i)
        times.append(perf_counter() - start)
        i+=1
        rate.sleep()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Saving times array.")
finally:
    np.save("times_3_Hz_non_blocking.npy", np.array(times))
    print("Times array saved.")

