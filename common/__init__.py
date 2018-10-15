import logging
import typing
import numpy as np
from tendo import colorer

logging.basicConfig(format="%(levelname)s - %(filename)s:%(lineno)s => %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def incremental_average(new_val: float, current_avg: float, n: int) -> float:
    return current_avg + (new_val - current_avg)/n


def incremental_weighted_average(new_val: float, current_avg: float, n: int, alpha: float) -> float:
    return (current_avg + new_val) * (1 - alpha)
