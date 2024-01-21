from scipy.spatial import distance
from collections import defaultdict
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from random import random
from random import randint
from random import choices
import numpy as np


class HLNSW(object):
  
  def l2_distance(self, a, b):
    return np.linalg.norm(a - b)

  def cosine_distance(self, a, b):
    try:
        return np.dot(a, b)/(np.linalg.norm(a)*(np.linalg.norm(b)))
    except ValueError:
        print(a)
        print(b)

  def _distance(self, x, y):
    return self.distance_func(x, [y])[0]

  def vectorized_distance_(self, x, ys):
    return [self.distance_func(x, y) for y in ys]

  def __init__(self,distance_type, m=16, ef=200, m0=None, heuristic=True, vectorized=False):
    if distance_type == "l2":
      # l2 distance
      distance_func = self.l2_distance
    elif distance_type == "cosine":
        # cosine distance
      distance_func = self.cosine_distance
    else:
      raise TypeError('Please check your distance type!')

    self.distance_func = distance_func

    if vectorized:
      self.distance = self._distance
      self.vectorized_distance = distance_func
    else:
      self.distance = distance_func
      self.vectorized_distance = self.vectorized_distance_
    self.data = {}
    self._m = m
    self._ef = ef
    self._m0 = 2 * m if m0 is None else m0
    self._level_mult = 1 / log2(m)
    self._graphs = []
    self._enter_point = None

    self._select = (
          self._select_heuristic if heuristic else self._select_naive)

  def add(self,tid,elem,ef=None):

    if ef is None:
      ef = self._ef
    distance = self.distance
    data = self.data
    graphs = self._graphs
    point = self._enter_point
    m = self._m


    # level at which the element will be inserted
    level = 1
    # print("Level: %d" % level)

    # elem will be at data[idx]
    idx = tid
    data[idx] = elem

    if point is not None:  
        # if the HLG is not empty, we have an entry point
        dist = distance(elem, data[point])
        # print(dist)
        # for all levels in which we dont have to insert elem,
        # we search for the closest neighbor
        for layer in reversed(graphs[level:]):
            point, dist = self._search_graph_ef1(elem, point, dist, layer)
        # at these levels we have to insert elem; ep is a heap of entry points.
        ep = [(-dist, point)]
        # print('EP: ',ep)
        layer0 = graphs[0]

        for layer in reversed(graphs[:level]):
          level_m = m if layer is not layer0 else self._m0
          # navigate the graph and update ep with the closest nodes we find
          ep = self._search_graph(elem, ep, layer, ef)
          # insert in g[idx] the best neighbors
          layer[idx] = layer_idx = {}
          self._select(layer_idx, ep, level_m, layer, heap=True)
          # assert len(layer_idx) <= level_m
          # insert backlinks to the new node
          for j, dist in layer_idx.items():
              self._select(layer[j], (idx, dist), level_m, layer)
              # assert len(g[j]) <= level_m
          # assert all(e in g for _, e in ep)
    
    for i in range(len(graphs), level):
        # for all new levels, we create an empty graph
        graphs.append({idx: {}})
        self._enter_point = idx
    # print(graphs)
    

  def _search_graph_ef1(self, q, entry, dist, layer):
    """Equivalent to _search_graph when ef=1."""

    vectorized_distance = self.vectorized_distance
    data = self.data

    best = entry
    best_dist = dist
    candidates = [(dist, entry)]
    visited = set([entry])
    while candidates:
      dist, c = heappop(candidates)
      # print('DIST TYPE:',type(dist),'DIST',dist)
      if dist > best_dist:
          break
      edges = [e for e in layer[c] if e not in visited]
      visited.update(edges)
      dists = vectorized_distance(q, [data[e] for e in edges])
      for e, dist in zip(edges, dists):
          
          if dist < best_dist:
              best = e
              best_dist = dist
              heappush(candidates, (dist, e))
              # break

    return best, best_dist

  def _search_graph(self, q, ep, layer, ef):

    vectorized_distance = self.vectorized_distance
    data = self.data

    candidates = [(-mdist, p) for mdist, p in ep]
    heapify(candidates)
    visited = set(p for _, p in ep)

    while candidates:
      dist, c = heappop(candidates)
      mref = ep[0][0]
      if dist > -mref:
        break
      # pprint.pprint(layer[c])
      edges = [e for e in layer[c] if e not in visited]
      visited.update(edges)
      dists = vectorized_distance(q, [data[e] for e in edges])
      for e, dist in zip(edges, dists):
        mdist = -dist
        if len(ep) < ef:
          heappush(candidates, (dist, e))
          heappush(ep, (mdist, e))
          mref = ep[0][0]
        elif mdist > mref:
          heappush(candidates, (dist, e))
          heapreplace(ep, (mdist, e))
          mref = ep[0][0]

    return ep


  def _select_heuristic(self, d, to_insert, m, g, heap=False):

    nb_dicts = [g[idx] for idx in d]

    def prioritize(idx, dist):
      return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx

    if not heap:
      idx, dist = to_insert
      to_insert = [prioritize(idx, dist)]
    else:
      to_insert = nsmallest(m, (prioritize(idx, -mdist)
                                for mdist, idx in to_insert))

    assert len(to_insert) > 0
    assert not any(idx in d for _, _, idx in to_insert)

    unchecked = m - len(d)
    assert 0 <= unchecked <= m
    to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
    to_check = len(checked_ins)
    if to_check > 0:
      checked_del = nlargest(to_check, (prioritize(idx, dist)
                                        for idx, dist in d.items()))
    else:
      checked_del = []
    for _, dist, idx in to_insert:
      d[idx] = dist
    zipped = zip(checked_ins, checked_del)
    for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
      if (p_old, d_old) <= (p_new, d_new):
          break
      del d[idx_old]
      d[idx_new] = d_new
      assert len(d) == m

  def search(self, q, k=None, ef=None):
    """Find the k points closest to q."""
    # print('q: ',type(q), ' k: ',type(k),' ef: ',type(ef))
    distance = self.distance
    graphs = self._graphs
    point = self._enter_point

    if ef is None:
      ef = self._ef

    if point is None:
      raise ValueError("Empty graph")
    # print('KKKKKK:',k)
    dist = distance(q, self.data[point])
    # look for the closest neighbor from the top to the 2nd level
    # for layer in reversed(graphs[1:]):
    #   point, dist = self._search_graph_ef1(q, point, dist, layer)
    # look for ef neighbors in the bottom level
    ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

    if k is not None:
      ep = nlargest(k, ep)
    else:
      ep.sort(reverse=True)

    return [(idx, -md) for md, idx in ep]