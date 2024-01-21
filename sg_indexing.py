import sg
import time
import filter_outlier
from collections import defaultdict
from math import log2
import pickle
import os

def build_index(name,path,vectors,M,ef):
  t = time.time()
  layered_dict,num_layers = layering(vectors,M)
  # print('Layering time: ',time.time()-t)
  # M = M - (num_layers//2)
  index = sg.HLNSW('l2', m = M - num_layers, ef=ef)
  t = time.time()
  for idx,v in enumerate(vectors):
    index.add(idx,v)
    # if idx%100==0:
    #   print('Added ',idx ,' points' )
  # print('Phase 1 time: ',time.time()-t)

  graphs = index._graphs
  level = len(graphs)
  # print('OLDDDDDDDDDDDDDDD: ',graphs)

  # print('Layered List:', layered_dict)
  layer_val_map = defaultdict(list)
  for k,v in layered_dict.items():
    layer_val_map[v].append(k)
  
  # print('Layer val map: ', layer_val_map)
  layer_indexes = []
  add_point_time = time.time()
  for layers in layer_val_map.keys():
    if layers != 1:
      layer = layer_val_map[layers]
      layer_index_name = name + '_layers'+ str(layers) + '.ind'
      layer_indexes.append(layer_index_name)
      layer_index = sg.HLNSW('l2', m = M - num_layers, ef=ef)
      for id in layer:
        layer_index.add(id,vectors[id])
      p = path + layer_index_name 
      
      with open(p, 'wb') as f:
        pickle.dump(layer_index, f, pickle.HIGHEST_PROTOCOL)
  # print("Layer indexes saving time: %f" % (time.time() - add_point_time))
    
  t = time.time()
  for j in layer_val_map.keys():
    for k in layer_val_map.keys():
      if j<k:
        # print('JJJJJJJJJJJJJJ: ',j,'KKKKKKKKKKKKKK: ',k)
        current = layer_val_map[j]
        # print(current)
        layer_index_name = name + '_layers'+ str(k) + '.ind'
        p = path + layer_index_name
        with open(p, 'rb') as f:
          l_index = pickle.load(f)
          # print('Loading layer index')
        for a in current:
          ngbrs = l_index.search(a,1)
          # print('Query:',a)
          
          for i in range(level):
            key = ngbrs[0][0]
            val = ngbrs[0][1]
            # insert = {key:val}
            # print(insert)
            graph = graphs[i]

            if a in graph.keys():
              # print("Inside if")
              d = graph[a]
              # print('Before adding: ',len(d))
              # print('Old:',d)
              if key not in d.keys():
                d[key] = val
                ng = {}
                ng[a]=val
                graph[key] = ng
          
              # print('New: ',d)
              # else:
              #   print("key already exists")
              # print('After adding: ',len(d))
          
  for li in layer_indexes:
     p = path + li
     os.remove(p)
  # print('Phase 2 time: ',time.time()-t) 
  print('Newwwwwwwwwwwww: ',graphs)  
  return index         
      
    

def layering(vectors,m):

  num_layers = int(log2(m))

  lower_bound, upper_bound, distances = filter_outlier.reject_outliers(vectors)

  bandwidth = upper_bound - lower_bound
  layer_width = bandwidth/num_layers
  layer_range = []
  for i in range(num_layers):
      layer_range.append(lower_bound + ((i+1)*layer_width))
  # print('Layer Range: ',layer_range)
  layer_dict = {}
  for idx,i in enumerate(distances):
      counter = 1
      for j in layer_range:
          if i>j:
              counter+=1
          layer_dict[idx] = counter

  return layer_dict,num_layers

def create_union_graph(graphs):
  union_graph = {}
  for graph in graphs:
      for node, neighbors in graph.items():
          if node not in union_graph:
              union_graph[node] = {}
          for neighbor, distance in neighbors.items():
              if neighbor not in union_graph:
                  union_graph[neighbor] = {}
              union_graph[node][neighbor] = distance
              union_graph[neighbor][node] = distance
  return union_graph