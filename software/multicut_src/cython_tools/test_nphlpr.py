import numpy as np
from numpy_helper import find_matching_indices_fast
import time

# return the indices of array which have at least one value from value list
def find_matching_indices(array, value_list):
    assert isinstance(array, np.ndarray)
    assert isinstance(value_list, np.ndarray) or isinstance(value_list, list)
    indices = []
    for i, row in enumerate(array):
        if( np.intersect1d(row, value_list).size ):
            indices.append(i)
    return np.array(indices)

x = np.random.randint(0, 1000, size = (100000,20), dtype = 'uint32' )
y = np.random.randint(0, 1000, size = (1000,),  dtype = 'uint32' )

N = 2

times = []
for _ in range(N):
    t0 = time.time()
    mask = find_matching_indices_fast(x,y)
    times.append(time.time()-t0)

print np.mean(times)
print

times = []
for _ in range(N):
    t0 = time.time()
    mask_np = find_matching_indices(x,y)
    times.append(time.time()-t0)

print "Pure np"
print np.mean(times)

assert mask.shape == mask_np.shape
assert np.all(mask_np == mask )
