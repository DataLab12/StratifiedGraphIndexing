import sys
import numpy as np
def ivecs_read(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 1)[:, 1:]
    vectors = data.tolist()
    return vectors

def fvecs_read(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 1)[:, 1:]
    vectors = data.tolist()
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors