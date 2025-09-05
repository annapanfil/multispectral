import time
from functools import wraps

def timer(timers, key, verbose=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start
            timers[key].append(elapsed)
            if verbose:
                print(f"Time of {key}: {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator

