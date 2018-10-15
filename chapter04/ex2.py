from common import logger
from common.environments.car_rental import CarRental

if __name__ == "__main__":

    dynamics = {}
    car_rental = CarRental(
        dynamics,
        max_c1=20,
        max_c2=20,
        s1_req_lambda=3,
        s1_ret_lambda=4,
        s2_req_lambda=2,
        s2_ret_lambda=3
    )

    print(car_rental.step(0, 0))
