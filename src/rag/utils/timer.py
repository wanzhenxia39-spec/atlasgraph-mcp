import time


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def cost(self) -> float:
        return time.time() - self.start

    def reset(self) -> None:
        self.start = time.time()
