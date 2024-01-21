# import numpy as np
# from scipy.optimize import linprog


# def in_hull(points, x):
#     n_points = len(points)
#     n_dim = len(x)
#     c = np.zeros(n_points)
#     A = np.r_[points.T,np.ones((1,n_points))]
#     b = np.r_[x, np.ones(1)]
#     lp = linprog(c, A_eq=A, b_eq=b)
#     return lp.success



# for i in Z:
#     print(in_hull(Z, i))

import numpy as np
import read_glove

def reject_outliers(data):

    u = np.mean(data,axis=0)
    # print('Mean: ',u)

    distances = []
    distances = np.linalg.norm(data - u[:], axis=1)
    distances.sort()
    
    # print(distances[-10:])
    std = np.std(distances)
    max_dist = max(distances)
    min_dist = min(distances)
    avg_dist = sum(distances)/len(distances)
    # distances_high = [d for d in distances if(d>avg_dist+3*std)]
    # print('Outliers: ',len(distances_high))
    lower_bound = min_dist
    upper_bound = avg_dist + 2*std
    # distances_high.sort()
    # print(distances_high[:10])
    # std_high = np.std(distances_high)
    print('Max: ',max_dist,' Min: ',min_dist, ' Avg:',avg_dist, ' Std all: ',std,' Exp:',avg_dist+2*std)
    
    return lower_bound,upper_bound,distances
# data_path = '/home/m_r1117/Desktop/Toufik/Practice/Indexing/Benchmark_Data/glove/glove.42B.300d.txt'
# data,dim = read_glove.get_embeddings(data_path)

# vectors = reject_outliers(data)

# print(vectors.shape)