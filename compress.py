import blosc
import numpy as np
import pickle
import gzip

data_path = '/home/m_r1117/Desktop/Toufik/Practice/Indexing/fbinFeatures/DIOR_features.fbin'
name = data_path.split('/')[-1][:-5]
idx_path = f"indexes/{name}_SG.ind"

with open(idx_path, 'rb') as f:
    index = pickle.load(f)
pickled_data = pickle.dumps(index)
# compressed_pickle = blosc.compress(pickled_data)

with gzip.open(idx_path, 'wb') as f:
    f.write(pickled_data)

with gzip.open(idx_path, 'rb') as f:
    print('Loading indexes...')
    compressed_pickle = f.read()
# depressed_pickle = blosc.decompress(compressed_pickle)
index = pickle.loads(compressed_pickle)