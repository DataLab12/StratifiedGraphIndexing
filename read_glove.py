import numpy as np

def get_embeddings(File):
    print("Loading Glove Data...")

    vec_list = []
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            # word = split_line[0]
            embedding = split_line[1:]
            if len(embedding)==300:
                vec_list.append(embedding)
            # if len(vec_list) >= 9999:
            #     break
                
        # print('Vec listttttttt',len(vec_list))
        vectors = np.asarray(vec_list,dtype=np.float32)
    return vectors, vectors.shape[1]

