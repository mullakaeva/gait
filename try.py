from common.utils import  OnlineFilter_np
import numpy as np


filter = OnlineFilter_np((3,), 5)

for i in range(10):
    if i < 5:
        foo = np.zeros(3) * np.nan
    else:
        foo = np.ones(3)
    filtered_output = filter.add(foo)

    print(filtered_output)