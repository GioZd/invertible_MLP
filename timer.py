import time

def timer(my_function):
    """Time your function with this decorator!"""
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = my_function(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"{my_function.__name__} ran in {t2 - t1:.3f} sec")
        return result
    return wrapper