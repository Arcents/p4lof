# -*- coding: utf-8 -*-

from heapq import heappush, heappop
from itertools import count
import networkx as nx
import myutil as ui

def single_source_dijkstra_ecmp(G, source, target=None,
                                     cutoff=None, weight='weight'):
    """Compute shortest paths and lengths in a weighted graph G.

    Uses Dijkstra's algorithm for shortest paths.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned

    weight: string
       Name of the attribute in the edge that represents the weight

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the list of all shortest paths from that node to the
       destination.

    Notes
    ---------
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    Based on single_source_dijkstra as in networkx version 1.9
    """
    if source == target:
        return ({source: 0}, {source: [source]})
    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    paths = {source: [[source]]}  # dictionary of paths
    seen = {source: 0}
    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        # for ignore,w,edgedata in G.edges_iter(v,data=True):
        # is about 30% slower than the following
        if G.is_multigraph():
            edata = []
            for w, keydata in G[v].items():
                minweight = min((dd.get(weight, 1)
                                 for k, dd in keydata.items()))
                edata.append((w, {weight: minweight}))
        else:
            edata = iter(G[v].items())

        for w, edgedata in edata:
            vw_dist = dist[v] + edgedata.get(weight, 1)
            if cutoff is not None:
                if vw_dist > cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist
                push(fringe, (vw_dist, next(c), w))
                paths[w] = []
                for p in paths[v]:
                    paths[w].append(p[:] + [w])
            elif vw_dist == seen[w]:
                for p in paths[v]:
                    paths[w].append(p[:] + [w])
    return (dist,paths)

def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def shortest_paths_dag(G, destination, pathfile, minlen):
    """Return the DiGraph of shortest paths from all nodes to the destination.

    Parameters
    ----------
    G : NetworkX graph

    destination : node label
        
    Returns
    -------
    NetworkX DiGraph, an unweighted DAG of all shortest paths to the
    destination node.
    """
    (dist, paths) = single_source_dijkstra_ecmp(G, destination)
    for i in dist.values():
        if i < minlen :
            k = get_key(dist, i)
            for j in k:        
                del dist[j]
                del paths[j]
    # print("max length of all paths is: %d"%max(dist.values()))
    # print("min length of all paths is: %d"%min(dist.values())) 
    # print("mean length of all paths is: %d"%(sum(dist.values())/len(dist.values())))
    
    dag = nx.DiGraph()

    for _, shortest in paths.items():
        for path in shortest:
            # single_source_dijkstra_ecmp gives us the paths *from* the destination
            # so we reverse them on the fly.  
            path.reverse()
            dag.add_path(path)

    with open(pathfile, 'w') as f:
        for key in paths:
            f.writelines(str(key) + ':' + str(paths[key]))
            f.write('\n')
    ui.resort_pathfile(pathfile)
    return dag

def shortest_paths_srcs_only(G, srcs, destination):
    """Return the DiGraph of shortest paths from all src nodes to the destination.

    Parameters
    ----------
    G : NetworkX graph
    srcs: src nodes
    destination : node label
    
    Returns
    -------
    NetworkX DiGraph
    """
    (dist, paths) = single_source_dijkstra_ecmp(G, destination)
    dag = nx.DiGraph()
    path_count = 0
    for _, shortest in paths.items():
        for path in shortest:
            # single_source_dijkstra_ecmp gives us the paths *from* the destination
            # so we reverse them on the fly.
            path.reverse()
            if (path[0] in srcs):            
                dag.add_path(path)
                path_count = path_count + 1
            else:
                break    
    print("num of all the paths from src nodes to dst: %d"%path_count)

    return dag