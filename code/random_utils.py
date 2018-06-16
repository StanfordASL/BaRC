from contextlib import contextmanager
import numpy as np
from random import seed, setstate, getstate


@contextmanager
def fixed_random_seed(seed_value):
    old_state = getstate()
    old_np_state = np.random.get_state()

    seed(seed_value)
    np.random.seed(seed_value)

    yield

    np.random.set_state(old_np_state)
    setstate(old_state)


if __name__ == '__main__':
    from random import randint
    seed(0)
    np.random.seed(0)

    print('1', randint(0, 1000), np.random.randint(low=0, high=1001))
    print('2', randint(0, 1000), np.random.randint(low=0, high=1001))
    print('3', randint(0, 1000), np.random.randint(low=0, high=1001))

    seed(0)
    np.random.seed(0)

    print('1', randint(0, 1000), np.random.randint(low=0, high=1001))
    print('2', randint(0, 1000), np.random.randint(low=0, high=1001))

    with fixed_random_seed(0):
        print('1', randint(0, 1000), np.random.randint(low=0, high=1001))
        with fixed_random_seed(0):
            print('1', randint(0, 1000), np.random.randint(low=0, high=1001))

        print('2', randint(0, 1000), np.random.randint(low=0, high=1001))

    print('3', randint(0, 1000), np.random.randint(low=0, high=1001))
