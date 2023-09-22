import numpy as np
from functools import wraps
from termcolor import colored


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))




def logging(enabled = True, message = "", color = "yellow"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                print(f"LOG: {colored(message, color = color)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator