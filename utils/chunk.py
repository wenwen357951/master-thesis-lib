from math import ceil


def chunks(lst: list, n_stack: int):
    total = len(lst)
    n = ceil(total / n_stack)
    for i in range(0, total, n):
        yield lst[i:i + n]
