import logging
import numpy as np
from tendo import colorer

logging.basicConfig(format="%(levelname)s - %(filename)s:%(lineno)s => %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def incremental_average(new, current_avg, n):
    return current_avg + (new - current_avg)/n


def incremental_weighted_average(new, current_avg, n, alpha):
    return (current_avg + new) * (1 - alpha)
