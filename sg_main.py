from random import random
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
import pickle
import read_DEEP
import os
from sklearn.metrics import precision_score
import evaluation
import read_SIFT
import read_glove
import sg_indexing
import gzip


M = 8
ef = 100




data_path = './sample_data/DEEP10k.fbin'



data,dim = read_DEEP.read_fbin(data_path)


quit_cmd = 'a'
while(quit_cmd != 'q'):
    n_neighbors = input('Enter the number of nearest neghbors: ')
    n_neighbors = int(n_neighbors)

    # query_indexes = [0,15,123,500,888,1234,3456,4567,7890,8888]

    # query_indexes = [10,20,500,3456,7890,12345,34567,66666,77777,88888]
    # query_indexes = [10,20,500,3456,7890,1234,34567,66666,77777,88888]
    query_indexes = [0]

    print('Data shape: ',data.shape)


    t = time.time()
    index_brute = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(data)
    print('Indexing time for Brute: ', time.time()-t,'seconds')

    # name = 'demo'
    name = data_path.split('/')[-1][:-5]
    path = f"indexes/"
    idx_path = f"indexes/{name}_SG.ind"
    if not os.path.exists(idx_path):
        add_point_time = time.time()
        # for idx, i in enumerate(data):
        #     index.add(i)
        #     # hnsw_balanced.add(i)
        print('Indexing...')
        index = sg_indexing.build_index(name,path,data,M,ef)
        print("SG indexing time: %f" % (time.time() - add_point_time))
        pickled_data = pickle.dumps(index)

        with gzip.open(idx_path, 'wb') as f:
            f.write(pickled_data)



    add_point_time = time.time()
    with gzip.open(idx_path, 'rb') as f:
        print('Loading indexes...')
        compressed_pickle = f.read()

    index = pickle.loads(compressed_pickle)
    print("SG index loading time: %f" % (time.time() - add_point_time))



    recall_final = 0
    precision_final = 0
    f1_score_final = 0
    total_search_time = 0

    for q in query_indexes:
        query = data[q]
        query = query[None,:]
        t = time.time()
        idx_brute = index_brute.kneighbors(query, n_neighbors, return_distance=False)[0]
        print('Searching time brute: = ', time.time()-t)
        true = idx_brute.tolist()
        # print('True IDX:',true)

        add_point_time = time.time()
        predicted = index.search(query, k=n_neighbors)
        search_time = time.time()
        single_query_time = search_time - add_point_time
        # print('Search time:',single_query_time)
        total_search_time = total_search_time + single_query_time
    
        pred = []
        for i in predicted:
            # print('IDX:',i[0])
            pred.append(i[0])
        # print('TRUEEEEEE: ',true)
        # print('PREDDDDDD: ',pred)

        recall = evaluation.recall(true,pred)
        print('Recall intermediate for ',q,' is: ',recall)
        recall_final = recall_final + recall
        precision = evaluation.precision(true,pred)
        precision_final = precision_final + precision
        print('Precision intermediate for ',q,' is: ',precision)

    rec = recall_final/len(query_indexes)
    pre = precision_final/len(query_indexes)
    f1_score = 2 * (pre* rec) / (pre+ rec)
    avg_search_time = total_search_time/len(query_indexes)
    print('Recall @',n_neighbors,' is: ',rec)
    print('Precision @',n_neighbors,' is: ',pre)
    print('F1-score @',n_neighbors,' is: ',f1_score)
    print("Searchtime SG: ", avg_search_time)

    quit_cmd = input('Enter q to quit: ')

