#!/usr/bin/env python

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
import random
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import wave
import math
import struct
import argparse
from itertools import *
from base64 import urlsafe_b64encode
import gzip
from io import BytesIO, StringIO
import numpy as np
import target1
from rdf import Triple, URIRef, Literal
import datetime
from datetime import timedelta

def symmetric_rels(g, symm):
    # set proportion of symmetric relations
    
    tups = []
    for s,o in g.edges:
        if s < o:
            tups.append((o,s))
        else:
            tups.append((s,o))
    cnts = Counter(tups)
    res = [key for key, val in cnts.items() if val == 2]
    
    for s,o in res:
        s,o = random.choice([(s,o),(o,s)])
        if g.out_degree(s)!=1 and g.in_degree(o)!=1:
            g.remove_edge(s,o)
        
    exclude = set()
    for i in range(int(math.ceil(len(g.edges()) * symm)/2)):
        rem = random.choice([e for e in g.edges() if e not in exclude])
        g.remove_edge(rem[0],rem[1])
        new = random.choice([(j,i) for i,j in g.edges()])
        g.add_edge(new[0],new[1])
        exclude.add((new[0],new[1]))
        exclude.add((new[1],new[0]))
    
    return g

def no_symmetric_rels(g):
    # set proportion of symmetric relations
    
    tups = []
    for s,o in g.edges:
        if s < o:
            tups.append((o,s))
        else:
            tups.append((s,o))
    cnts = Counter(tups)
    res = [key for key, val in cnts.items() if val == 2]
    
    for s,o in res:
        if g.degree(s)>1 and g.degree(o)>1:
            g.remove_edge(s,o)
    
    return g
  
def kg_density(g):
    # need this to calculate for directed graph
    
    num_edges = len(g.edges)
    num_nodes = len(g.nodes)
    density = num_edges / (num_nodes *(num_nodes - 1))
    
    return density

def add_link(g, source):
    # add a link that doesn't already exist
    
    link = False
    while link == False:
        target = random.choice([i for i in g.nodes()])
        if g.has_edge(source,target):
            continue
        else:
            g.add_edge(source,target)
            link = True
            
    return

def graph_density(n_nodes, density):
    # keep removing or adding edges based on desired density
    
    print(f'Starting density {density}')
    
    # if desired density is too low for initial graph generation
    if int(n_nodes * density) < 2:
        neighbors = 2
        g = nx.connected_watts_strogatz_graph(n=n_nodes, k=neighbors, p=0.2, tries=1000)
        real_density = kg_density(g)
        count = 0
        if real_density - density > 0:
            print('decreasing density')
            while real_density - density > 0:
                if count > n_nodes*10:
                    break
                # can change range for efficiency at expense of accuracy
                for i in range(1):
                    source = random.choice(list(g.nodes))
                    edges = [n for n in g.edges(source)]
                    if len(edges) >= 1:
                        target = random.choice(edges)
                        target = target[1]
                        if source > target:
                            g.remove_edge(target,source)
                        else:
                            g.remove_edge(source,target)
                        count += 1
                        if count%1000==0:
                            print(f'round {count}: {real_density}')
                real_density = kg_density(g)
        else:
            count=0
            if real_density - density < 0:
                print('increasing density')
                while real_density - density < 0:
                    if count > n_nodes*100:
                        break
                    for i in range(1):
                        source = random.choice(list(g.nodes))
                        add_link(g, source)
                        real_density = kg_density(g)
                        count += 1
                        if count%1000==0:
                            print(f'round {count}: {real_density}')
            
    else:
        neighbors = int(n_nodes * density)*2
        g = nx.connected_watts_strogatz_graph(n=n_nodes, k=neighbors, p=0.2, tries=1000)
        real_density = kg_density(g)
        count=0
        if real_density - density < 0:
            print('increasing density')
            while real_density - density < 0:
                if count > n_nodes*100:
                    break
                for i in range(1):
                    source = random.choice(list(g.nodes))
                    add_link(g, source)
                    real_density = kg_density(g)
                    count += 1
                    if count%1000==0:
                        print(f'round {count}: {real_density}')
        else:
            print('decreasing density')
            while real_density - density > 0:
                if count > n_nodes*10:
                    break
                # can change range for efficiency at expense of accuracy
                for i in range(1):
                    source = random.choice(list(g.nodes))
                    edges = [n for n in g.edges(source)]
                    if len(edges) >= 1:
                        target = random.choice(edges)
                        target = target[1]
                        if source > target:
                            g.remove_edge(target,source)
                        else:
                            g.remove_edge(source,target)
                        count += 1
                        if count%50==0:
                            print(f'round {count}: {real_density}')
                real_density = kg_density(g)
    print(f'Final density: {real_density}')
    
    return g


def calc_in_deg(g, deg):
    # calculate ratio of nodes that have a given in-degree
    
    count = 0
    for i in g.nodes():
        in_d = g.in_degree(i)
        if in_d == deg:
            count+=1
    ratio = count / len(g.nodes())
    
    return ratio

def calc_reflex(g):
    # calculate ratio of nodes that have a reflexive relation
    
    count = 0
    for i in g.nodes:
        edges = g.edges(i)
        for j in edges:
            if i == j[1]:
                count+=1
    reflexivity_rate = (count/(len(g.nodes)))
    
    return reflexivity_rate

def calc_symm(g):
    # calculate ratio of relations that are symmetric
    
    symm_set = set()
    
    for i,j in g.edges():
        if g.has_edge(i,j) and g.has_edge(j,i) and i!=j:
            symm_set.add((i,j))
    symm_ratio = len(symm_set)/len(g.edges())
    
    return symm_ratio, symm_set
            
                    
def calc_out_deg(g, deg):
    # calculate ratio of nodes that have a given out-degree

    count = 0
    for i in g.nodes():
        out_d = g.out_degree(i)
        if out_d == deg:
            count+=1
    ratio = count / len(g.nodes())
    
    return ratio

def reflexive_nodes(g):
    # create list of nodes that are reflexive (not really used)
    
    reflex_list = []
    
    for i in g.nodes:
        edges = g.edges(i)
        for j in edges:
            if i == j[1]:
                reflex_list.append((i,i))
    
    return reflex_list

def calc_trans(g,hops):
    # calculate ratio of nodes with hops of a given number to another node 
    
    trans_nodes = set()
    minus_trans = set()
    plus_trans = set()
    node_pairs = 0
    trans_pairs = 0

    for i in g.nodes():
        for j in g.nodes():
            all_paths = nx.all_simple_paths(g,i,j, cutoff = hops+1)
            d_n = [n for n in all_paths if len(n)>0]
            good = [n for n in d_n if len(n)==hops+1]
            bad = [n for n in d_n if len(n)!=2 and len(n)!=hops+1]
            if len(bad) == 0 and len(good)!=0:
                trans_pairs+=1
                path = random.choice(good)
                for k in range(hops):
                    trans_nodes.add((path[k],path[k+1]))
            elif len(bad)>0:
                minus_hops = [n for n in bad if len(n) == hops]
                plus_hops = [n for n in bad if len(n) == hops+2]
                for n in plus_hops:
                    plus_trans.add((n[0],n[1]))
                for n in minus_hops:
                    minus_trans.add(n[-1])
            else:
                continue
            node_pairs += 1
    
    existing_ratio = trans_pairs / (node_pairs)
    
    
    return existing_ratio, trans_nodes, minus_trans, plus_trans

def edge_not_in_tri(g):
    # find all nodes not in triangle, used for adjusting clustering coefficient
    
    triangle_nodes = set()
    triangles = []
    # Iterate over all possible triangle relationship combinations
    for n in g.nodes():
        for n1, n2 in combinations(g.neighbors(n), 2):
            # Check if n1 and n2 have an edge between them, add to trianlge_nodes if yes
            if g.has_edge(n1, n2):
                triangle_nodes.add((n,n1))
                triangle_nodes.add((n,n2))
                triangle_nodes.add((n1,n2))
            
    not_tri = set()
    for i in triangle_nodes:
        if len(i)>0:
            triangles.append(i)
    edges = [n for n in g.edges() if n not in triangles]
    
    return edges


def nodes_in_triangle(graph, n):
    # find all triangles for a given node, used for adjusting clustering coefficient
    
    triangle_nodes = set()
    triangles = []
    neighbors = [i for i in graph.neighbors(n) if i!=n]
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(neighbors, 2):
        # Check if n1 and n2 have an edge between them
        if graph.has_edge(n1, n2):
            triangle_nodes.add((n1,n2))
            triangle_nodes.add((n,n1))
            triangle_nodes.add((n,n2))
        
    for i in triangle_nodes:
        if len(i)>0:
            triangles.append(i)
    non_triangles = [n for n in graph.edges() if n not in triangles]
    
    return triangles, non_triangles

def make_directed(g):
    # convert to directed graph by randomly assigning direction
    
    tups = []
    for s,o in g.edges:
        if s < o:
            tups.append((o,s))
        else:
            tups.append((s,o))
    cnts = Counter(tups)
    res = [key for key, val in cnts.items() if val == 2]
    
    for s,o in res:
        s,o = random.choice([(s,o),(o,s)])
        g.remove_edge(s,o)
        
    return g    

def avg_clustering(g, clustering):
    # keep removing or adding triangles based on desired average clustering coefficient
    
    reflex_edges = reflexive_nodes(g)
    count = 0
    real_clustering = nx.average_clustering(g)
    tris = []
    print(f'starting clust {real_clustering}')
    if real_clustering - clustering > 0:
        print('decreasing clustering')
        while real_clustering - clustering > 0:
            if count > 1000:
                break
            for i in range(10):
                while len(tris) < 1:
                    head = random.choice([n for n in g.nodes() if g.degree(n)>1]) 
                    tri, non_tri = nodes_in_triangle(g,head)
                    if len(tri)>0:
                        tris.append((tri))
                tails = [n[1] for n in tris[0] if n[0]!=head and g.in_degree(n[1])>1]
                if len(tails) > 0:
                    symm_ratio, symm_set = calc_symm(g)
                    pos_tails = [n for n in tails if (head,n) not in symm_set]
                    if len(pos_tails)>0:
                        tail = random.choice(pos_tails)
                        g.remove_edge(head,tail)
                        new = False
                        c = 0
                        while new == False:
                            if c > 10:
                                break
                            rand = random.choice([n for n in g.nodes()]) 
                            tri, non_tri = nodes_in_triangle(g,rand)
                            if len(non_tri) > 0:
                                tailrs = random.choice(non_tri)
                                tailr = tailrs[0]
                                neighbors = [n[1] for n in g.edges(head)]
                                excl = neighbors + [head, tailr]
                                tailx = random.choice([w for w in g.nodes() if w not in excl])
                                if (not(g.has_edge(tailr,tailx))) and (not(g.has_edge(tailx,tailr))) and (not(g.has_edge(tailx,head))):
                                    g.add_edge(head,tailx)
                                    new = True
                            c +=1
                tris = []
            real_clustering = nx.average_clustering(g) 
            count += 1
    else:
        print('increasing clustering')
        while real_clustering - clustering < 0:
            if count > 1000:
                break
            for i in range(10):
                source = random.choice([n for n in g.nodes() if g.degree(n)>1])
                edges = [n for n in g.edges()]
                in_nodes = [n[0] for n in g.in_edges(source)]
                out_nodes = [n for n in g.neighbors(source)]
                connected = in_nodes + out_nodes
                if len(connected) > 0:
                    new_h = random.choice(connected)
                    potentials = [n for n in connected if n!=new_h and (n,new_h) not in edges 
                                  and (new_h,n) not in edges]
                    if len(potentials)>0:
                        new_t = random.choice(potentials)
                        g.add_edge(new_h,new_t)
            not_tris = edge_not_in_tri(g)
            symm_ratio, symm_set = calc_symm(g) 
            exclude = set()
            for i in range(10):
                pots = [n for n in not_tris if n[0]!=n[1] and g.degree(n[0])>1 and g.degree(n[1])>1 
                        and n not in exclude and n not in symm_set]
                if len(pots)>0:
                    rem_e = random.choice(pots)
                    g.remove_edge(rem_e[0], rem_e[1])
                    exclude.add(rem_e)
            real_clustering = nx.average_clustering(g)
            if count%50==0:
                print(f'round {count}: {real_clustering}')
            count += 1
    print(f'final average clustering coefficient {real_clustering}')
    
    return g

    
def avg_shortest(g):
    # calculate the average length of all simple paths in graph; doesn't scale well
    
    x = [n for n in nx.shortest_path_length(g)]
    lengths = 0
    total_paths = 0
    for i in x:
        lengths+=sum(i[1].values())
        total_paths+= len(i[1])
        avg = (lengths/total_paths)
        
    return avg

def remove_shortest(g, path_len, in_deg, out_deg):
    # remove the shortest existing path from the graph
    
    shortest = []
    node = random.choice([n for n in g.nodes()])
    v = [n for n in nx.descendants(g,node)]
    for i in v:
        w = [n for n in nx.all_shortest_paths(g,node,i)]
        z = min(w, key=len)
        shortest.append(z)
    y = [n for n in shortest if len(n)>2]
    if len(y)>0:
        q = min(y, key=len)
        if len(q)<= path_len:
            for i in range(len(q)-1):
                if g.out_degree(q[i])!=out_deg and g.in_degree(q[i+1])!=in_deg:
                    g.remove_edge(q[i],q[i+1])

def all_shortest(g,out_deg):
    # calculate all shortest simple paths; doesn't scale with graph size
    
    shortest = []
    node = random.choice([n for n in g.nodes()])
    v = [n for n in nx.descendants(g,node)]
    for i in v:
        w = [n for n in nx.all_shortest_paths(g,node,i) if g.out_degree(n[-1])!=out_deg 
             and (out_deg==None or g.out_degree(n[-1])!=out_deg -1)]
        if len(w)>0:
            z = min(w, key=len)
            shortest.append(z)
            
    return shortest

def avg_path(g, path_len, in_deg, out_deg): 
    # increase or decrease the average length of all shortest simplest paths in the graph
    
    avg_len = avg_shortest(g)
    tot_count = 0
    print(f'starting average shortest path length {avg_len}')

    if avg_len - path_len > 0:
        print('decreasing path')
        while (avg_len - path_len > 0):
            if tot_count > (100):
                break
            
            for i in range(10):
                head = random.choice([n for n in g.nodes() if g.out_degree(n)!=out_deg 
                                      and (out_deg==None or g.out_degree(n)!=out_deg-1)])
                descendants = [n for n in nx.descendants(g,head) if n!= head and 
                               g.in_degree(n)!=in_deg and (in_deg == None or g.in_degree(n)!=in_deg-1) 
                               and (n,head) not in g.edges()]
                if len(descendants) > 0:
                    tail = random.choice(descendants)
                    paths = list(nx.all_shortest_paths(g,head,tail))
                    path = random.choice(paths)
                    if len(path) > path_len:
                        g.add_edge(head,path[-1])
            
            avg_len = avg_shortest(g)
            tot_count += 1
    else:
        print('increasing path')
        while (avg_len - path_len < 0):
            if tot_count > (1000):
                break
            paths = all_shortest(g,out_deg)
            if len(paths)>0:
                ps = [n for n in paths if g.out_degree(n[-1])!=out_deg and (out_deg==None or g.out_degree(n[-1])!=out_deg-1)]
                if len(ps) > 0:
                    path = random.choice(ps)
                    ext_head = path[-1]
                    exclude = list(nx.descendants(g,ext_head))
                    ext_tails = [n for n in g.nodes() if n not in exclude and g.in_degree(n)!=in_deg 
                                 and (in_deg==None or g.in_degree(n)!=in_deg-1) and (n,ext_head) not in g.edges() 
                                 and n!=ext_head]
                    if len(ext_tails) > 0:
                        ext_tail=random.choice(ext_tails)
                        g.add_edge(ext_head,ext_tail)
                    remove_shortest(g,path_len, in_deg, out_deg)
                    
                    avg_len = avg_shortest(g)
            tot_count += 1

    print(f'final average shortest path length {avg_len}')   
    
    return g

def transitivity(g, hops, ratio, in_deg, out_deg):
    # keep removing or adding edges based on desired hops and ratio
    
    info = calc_trans(g,hops)
    current_ratio = info[0]
    i_hops = info[1]
    minus_hops = info[2]
    plus_hops = info[3]
    count = 0
    print(f'Starting {hops}-hop transitivity ratio: {current_ratio}')

    if current_ratio < ratio:
        print('increasing transitivity')
        while current_ratio - ratio < 0:
            if count > 1000:
                break
            if len(minus_hops) < 1 and len(plus_hops) < 1:
                break
            symm_ratio, symm_set = calc_symm(g)
            chance = random.random()
            if chance > 0.5:
                # remove edge 
                if len(plus_hops) > 0:
                    rem_es = [n for n in plus_hops if n not in symm_set and g.degree(n[0])>1 and g.degree(n[1])>1
                                           and n not in i_hops and g.out_degree(n[0])!=out_deg and
                                           (out_deg==None or g.out_degree(n[0])!=out_deg+1) and 
                                           g.in_degree(n[1])!=in_deg and 
                                           (in_deg==None or g.in_degree(n[1])!=in_deg+1)]
                    if len(rem_es)>0:
                        rem_e = random.choice(rem_es)
                        g.remove_edge(rem_e[0],rem_e[1])
            else:
                # add edge
                edges = [n for n in g.edges()]
                head = random.choice([n for n in minus_hops if g.out_degree!=out_deg and 
                                      (out_deg==None or g.out_degree(n)!=out_deg-1)])
                tails = [n for n in g.nodes() if n!= head and g.in_degree(n)!=in_deg and 
                         (in_deg==None or g.in_degree(n)!=in_deg-1) and (n,head) not in edges 
                         and n not in nx.descendants(g,head)]
                if len(tails)>0:
                    tail = random.choice(tails)
                    g.add_edge(head,tail)
            info = calc_trans(g,hops)
            current_ratio = info[0]
            i_hops = info[1]
            minus_hops = info[2]
            plus_hops = info[3]
            count+=1                
    else:
        print('decreasing transitivity')
        while current_ratio - ratio > 0:
            if count > 1000:
                break
                        
            potentials = [n for n in i_hops if g.in_degree(n[1])!=in_deg and g.degree(n[0])>1 and g.degree(n[1])>1 and
                          (in_deg==None or g.in_degree(n[1])!=in_deg+1) and
                          g.out_degree(n[0])!=out_deg and (out_deg==None or g.out_degree(n[0])!=out_deg+1)]
            if len(potentials)>0:
                remove_e = random.choice(potentials)
                g.remove_edge(remove_e[0], remove_e[1])
            info = calc_trans(g,hops)
            current_ratio = info[0]
            i_hops = info[1]
            count+=1
            
    final = calc_trans(g,hops)
    print(f'Final ratio of node pairs with {hops}-Hops: {final[0]}')
    
    return g

def reflexivity(g, ratio):
    # keep removing or adding edges based on desired density
    
    reflexivity_rate = calc_reflex(g)
    print(f'starting reflex ratio: {reflexivity_rate}')
    symm_ratio, symm_set = calc_symm(g)
    
    count = 0
    while reflexivity_rate < ratio:
        if count>1000:
            break
        source = random.choice(list(g.nodes))
        edges = [n for n in g.edges(source)]
        if ((source,source) not in edges) and (len(edges) > 1):
            target = random.choice([n for n in g.edges() if n[0]!=n[1] and
                                    g.degree(n[0])>1 and g.degree(n[1])>1 and n not in symm_set])
            g.add_edge(source, source)           
            if len(target) > 0:
                g.remove_edge(target[0], target[1])
            reflexivity_rate = calc_reflex(g)
        count+=1
    print(f'final reflexivity ratio: {reflexivity_rate}')
    
    return g

def avg_in_degree(g,in_deg,ratio):
    # set the ratio of nodes with a given in_degree; "Destination hubs"
    
    actual_ratio = calc_in_deg(g,in_deg)
    count = 0
    reflexive_ns = reflexive_nodes(g)
    
    print(f'starting: {actual_ratio}')
    if actual_ratio - ratio < 0:
        print('increasing in_degree')
        while actual_ratio - ratio < 0:
            if count > 1000:
                break
            nodes = [n for n in g.nodes() if g.in_degree(n) != in_deg]
            node = random.choice(nodes)
            node_in = g.in_degree(node)
            diff =  node_in - in_deg
            if diff < 0:
                symm_ratio, symm_set = calc_symm(g)
                neighbors = g.in_edges(node)
                ns = [n[0] for n in neighbors] # all incoming nodes
                potential_ns = [n for n in g.nodes() if n not in ns and n!=node] 
                targets_add = [n for n in potential_ns if (n,node) not in g.edges()]
                if len(targets_add) >= abs(diff):
                    targets = np.random.choice(targets_add, size=abs(diff), replace=False)
                    for n in targets:
                        g.add_edge(n, node)
                        e_remove = [n for n in g.in_edges() if n[0]!=n[1] and n not in symm_set and 
                                    g.in_degree(n[1])!= in_deg and g.in_degree(n[1])!= in_deg+1]
                        if len(e_remove) > 0:
                            remove = random.choice(e_remove)
                            g.remove_edge(remove[0],remove[1])
                                                      
            elif diff > 0:
                    symm_ratio, symm_set = calc_symm(g)
                    incoming = [n[0] for n in g.in_edges(node) if n[0]!=n[1] and n not in symm_set]
                    if len(incoming) >= abs(diff):
                        targets = np.random.choice(incoming, size=abs(diff), replace=False)
                        for n in targets:
                            g.remove_edge(n,node)
                            new_tail = [n for n in g.nodes() if n!=node and g.in_degree(n)!= in_deg and 
                                        g.in_degree(n)!=in_deg-1 and (n,node) not in symm_set]
                            if len(new_tail)>0:
                                new_t = random.choice(new_tail)
                                g.add_edge(node,new_t)
            actual_ratio = calc_in_deg(g,in_deg)
            count += 1        
            
            
    if actual_ratio - ratio > 0:
        print('decreasing in_degree')
        while actual_ratio - ratio > 0:
            if count > 1000:
                break
            existing = [n for n in g.nodes() if g.in_degree(n) == in_deg]
            if len(existing)> 0 :
                node = random.choice(existing)
                symm_ratio, symm_set = calc_symm(g)
                in_ds = [n for n in g.in_edges(node) if n[0] != n[1] and n not in symm_set]
                chance = random.random()
                if chance > 0.5:
                    remove_e = random.choice(in_ds)
                    g.remove_edge(remove_e[0], remove_e[1])
                else:
                    exclude = [n for n in g.nodes() if n not in existing and n!=node and (node,n) not in symm_set]
                    new_in = random.choice(exclude)
                    g.add_edge(new_in,node)
                actual_ratio = calc_in_deg(g,in_deg)
            count += 1
            
    final = calc_in_deg(g,in_deg)
    print(f'final in-degree ratio: {final}')
    
    return g

def avg_out_degree(g,out_deg,in_deg,ratio):
    # set the ratio of nodes with a given out_degree; "Source hubs"

    actual_ratio = calc_out_deg(g,out_deg)
    count = 0
    
    print(f'starting: {actual_ratio}')
    if actual_ratio - ratio < 0:
        print('increasing out_degree')
        while actual_ratio - ratio < 0:
            if count > 1000:
                break
            nodes = [n for n in g.nodes() if g.out_degree(n) != out_deg]
            node = random.choice(nodes)
            node_out = g.out_degree(node)
            diff =  node_out - out_deg
            symm_ratio, symm_set = calc_symm(g)

            if diff < 0:
                neighbors = [n for n in g.neighbors(node)]
                potentials = [n for n in g.nodes() if n!=node and n not in neighbors and
                              g.in_degree(n)!=in_deg and (in_deg==None or g.in_degree(n)!=in_deg-1) 
                              and (n,node) not in g.edges()]
                targets = np.random.choice(potentials, size=abs(diff), replace=False)
                for i in targets:
                    g.add_edge(node,i)
                for i in range(len(targets)):
                    rem_es = [n for n in g.edges() if g.in_degree(n[1])!=in_deg 
                              and (in_deg==None or g.in_degree(n[1])!=in_deg+1)
                             and g.out_degree(n[0])!=in_deg and (out_deg==None or g.out_degree(n[0])!=out_deg+1) 
                             and n[0]!=n[1] and n not in symm_set]
                    if len(rem_es)>0:
                        rem_e = random.choice(rem_es)
                        g.remove_edge(rem_e[0],rem_e[1])
                
            elif diff > 0:
                neighbors = [n[1] for n in g.edges(node) if n[0]!=n[1] and 
                             g.in_degree(n[1])!=in_deg and (in_deg==None or g.in_degree(n[1])!=in_deg+1)
                             and n not in symm_set]
                if len(neighbors) >=diff:
                    targets = np.random.choice(neighbors, size=abs(diff), replace=False)
                    for i in targets:
                        g.remove_edge(node,i)
                    for i in range(len(targets)):
                        new_hs = [n for n in g.nodes() if g.out_degree(n)!=out_deg 
                                 and (out_deg==None or g.out_degree(n)!=out_deg-1)]
                        if len(new_hs) > 0:
                            new_h = random.choice(new_hs)
                            new_ts = [n for n in g.nodes() if n!=new_h and g.in_degree(n)!=in_deg 
                                     and (in_deg==None or g.in_degree(n)!=in_deg-1) and (n,new_h) not in symm_set]
                            if len(new_ts) > 0:
                                new_t = random.choice(new_ts)
                                g.add_edge(new_h,new_t)    
            else:
                break
            actual_ratio = calc_out_deg(g,out_deg)
            count += 1        
            
    if actual_ratio - ratio > 0:
        print('decreasing out_degree')
        while actual_ratio - ratio > 0:
            if count > 1000:
                break
            node = random.choice([n for n in g.nodes() if g.out_degree(n) == out_deg])
            chance = random.random()
            symm_ratio, symm_set = calc_symm(g)

            if chance > 0.5:
                symm_ratio, symm_set = calc_symm(g)
                rem_es = [n for n in g.edges(node) if n[0]!=n[1] and g.in_degree(n[1])!=in_deg 
                          and (in_deg==None or g.in_degree(n[1])!=in_deg+1) and n not in symm_set]
                if len(rem_es)>0:
                    eg = random.choice(rem_es)
                    g.remove_edge(eg[0], eg[1])
            else:
                target = random.choice([n for n in g.nodes() if (g.in_degree(n)!=in_deg) 
                                        and (in_deg==None or g.in_degree(n)!=in_deg-1) 
                                        and (n!=node) and (n,node) not in symm_set])
                g.add_edge(node,target)
            actual_ratio = calc_out_deg(g,out_deg)
            count += 1
                   
    final = calc_out_deg(g,out_deg)
    print(f'final out-degree ratio: {final}')
    

    return g

def longest_path(g,i,j):
    # get longest simple path between two nodes
    
    longest_len = 0
    longest_p = []
    paths = list(nx.all_simple_paths(g,i,j))
    if len(paths) > 0:
        longest_p = max(paths, key=len)
        longest_len = len(longest_p)
        
    return (i,longest_p), longest_len

def calc_avg_long_path(g):
    # calculate the average length of all the longest simple paths in the graph; doesn't scale
    
    longest_paths = []
    count = 0
    tot = 0
    zeros = 0
    for i in g.nodes():
        lp = 0
        long_p = []
        for j in g.nodes():
            longest_p, longest_len = longest_path(g,i,j)
            if longest_len > 0:
                count += longest_len
                tot+=1
                if longest_len > lp:
                    lp = longest_len
                    long_p = longest_p
            else:
                zeros +=1
        if lp == 0:
            long_p = (i,[])
        longest_paths.append(long_p)
    avg = count/tot
    
    return avg, longest_paths

def fewest_desc(nodes):
    fewest = len(g.nodes())
    target = int()
    for i in nodes:
        descs = list(nx.descendants(g,i))
        length = len(descs)
        if length < fewest:
            fewest = length
            target = i
            
    return target

def avg_longest_path(g, avg):
    # set the average longest path length, this function does not scale with KG size
    
    
    info = calc_avg_long_path(g)
    actual = info[0]
    grap = [n for n in g.edges()]
    
    count = 0
    if actual - avg < 0:
        print('increasing average longest path')
        while actual - avg < 0:
            if count >= 100:
                break
            
            node = random.choice([n for n in g.nodes()])
            path = [i[1] for i in info[1] if i[0]==node][0]
            if 0 < len(path) < len(g.nodes()):
                source = path[-1]
                potentials = [n for n in g.nodes() if n not in path]
                p = fewest_desc(potentials)
                g.add_edge(source,p)
            count += 1
            info = calc_avg_long_path(g)
            actual = info[0]
                
    else:
        print('decreasing average longest path')
        while actual - avg > 0:
            if count >= 100:
                break
            for b in range(5):
                node = random.choice([n for n in g.nodes()])
                path = [i[1] for i in info[1] if i[0]==node][0]
                p_len = len(path)
                if (3 < p_len):
                    head = path[p_len//2 -1]
                    tail = path[(p_len//2)]
                    g.remove_edge(head,tail)
                    info = calc_avg_long_path(g)
                    actual = info[0]
                count+=1
            
    return g

def shorten_dmtr(g, dmtr):
    # decrease the longest simple path length if it is greater than the desired value
    # increasing the longest simple path length not considered, would be trivial
    
    diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(g)])
    print(f'starting longest shortest simple path length {diameter}')
    count = 0
    while diameter > dmtr:
        head = random.choice([n for n in g.nodes()])
        tail = random.choice([n for n in g.nodes() if n!=head])
        if nx.has_path(g,head,tail):
            long = max([x for x in nx.all_shortest_paths(g,head,tail)], key=len)
            if len(long) > dmtr:
                y = long[:dmtr-1]
                z = long[dmtr-1:]
                g.add_edge(y[-1],z[-1])
                diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(g)])
        count+=1
    diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(g)])
    print(f'final longest shortest simple path length {diameter}')
    return g

def calc_deg(g):
    total = 0
    for n in g.nodes():
        total+= g.degree(n)
    avg = total / len(g.nodes())
    
    return avg

def gen_structure(n_nodes, density=None, symm=None, reflex=None,
             in_deg=None, in_ratio=None,
             out_deg=None, out_ratio=None,
             n_hops=None, hops_ratio=None,
             clustering=None, 
             diameter=None,
             avg_short_path=None, avg_long_path=None):
    
    if density != None:
        g = graph_density(n_nodes, density)
        print('density done')
        g = g.to_directed()
    else:
        g = nx.connected_watts_strogatz_graph(n_nodes, k=4, p=0.2)
        g = g.to_directed()
        g = make_directed(g)
    
    if symm!=None:
        g = symmetric_rels(g,symm)
        print(f'symm {calc_symm(g)[0]}')
    else:
        g = no_symmetric_rels(g)
    
    if reflex!=None:
        g = reflexivity(g,reflex)
        print('reflex done')
    
    if in_deg!=None:
        g = avg_in_degree(g,in_deg,in_ratio)
        print('in-degree done')
    
    if out_deg!=None:
        g = avg_out_degree(g,out_deg,in_deg,out_ratio)
        print('out-degree done')

    if avg_short_path!=None:
        g = avg_path(g, avg_short_path, in_deg, out_deg)
        print('average shortest path done')

    if diameter!=None:
        g = shorten_dmtr(g, diameter)
        diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(g)])
        print(f'longest path done')
    
    if n_hops!=None and hops_ratio!=None:
        g = transitivity(g,n_hops,hops_ratio,in_deg,out_deg)
        print('k-hops done')
        
    if clustering!=None:
        g = avg_clustering(g, clustering)
        print('clustering done')
    
    print(f'isolates {[n for n in nx.isolates(g)]}')
    print(f'final density {kg_density(g)}')
    diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(g)])
    print(f'longest shortest simple path length {diameter}')
    print(f'in degree {calc_in_deg(g,in_deg)}')
    print(f'out degree {calc_out_deg(g,out_deg)}')
    print(f'symm {calc_symm(g)[0]}')
    print(f'reflex {calc_reflex(g)}')
    print(f'final edges {len(g.edges)}')
    print(f'final average degree {calc_deg(g)}')
    print(f'clustering {nx.average_clustering(g)}')
    
    # doesnt' scale; uncomment print statement
    print(f'avg shortest path {avg_shortest(g)}')

        
    return g, g.edges()

# audio generation adapted from the works by Zach Denton: https://zach.se/generate-audio-with-python/
def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5):
    # generate repeating sine waves but use a lookup table instead of recalculating each time

    period = int(framerate / frequency)
    if amplitude > 1.0: amplitude = 1.0
    if amplitude < 0.0: amplitude = 0.0
    lookup_table = [float(amplitude) * math.sin(2.0*math.pi*float(frequency)*(float(i%period)/float(framerate))) for i in range(period)]
    return (lookup_table[i%period] for i in count(0))


def white_noise(amplitude=0.5):
    # generate random white noise
    
    return (float(amplitude) * random.uniform(-1, 1) for _ in count(0))

def compute_samples(channels, nsamples=None):
    # combining channels to get samples
    
    return islice(zip(*(map(sum, zip(*channel)) for channel in channels)), nsamples)

def gen_audio(framerate, amplitude, nchannels, nsamples=None):
    # generate 1 second audio clip (can extend for longer clips)
    
    freqs = []
    channels = []
    for i in range(nchannels):
        frequency = random.randint(1501,20000)
        freqs.append(frequency)
        channels.append((sine_wave(frequency, amplitude=0.1), white_noise(amplitude=0.001)))
    channels = tuple(channels)    
    samples = compute_samples(channels)
    
    count = 0
    wav = []
    max_amplitude = 32767

    for sample in samples:
        if count> (framerate - 1):
            break
        x = struct.pack('h',int(sample[0] * max_amplitude))
        y = struct.pack('h',int(sample[1] * max_amplitude))
        samp = b''.join([x,y])
        wav.append(samp)
        count += 1
        
    waves = b''.join(wav)
    return waves

def gen_string(vocab, min_length=5, max_length=30):
    length = np.random.randint(min_length, max_length)
    return ' '.join(np.random.choice(vocab, size=length))

def gen_anyURI(vocab, min_length=2, max_length=8):
    base_length = np.random.randint(min_length, max_length-1)
    path_length = np.random.randint(1, (max_length-base_length)+1)
    prefix = np.random.choice(['http', 'https', 'ftp', 'ssh', 'nfs'], 1)[0]
    return prefix + '://' + '.'.join(np.random.choice(vocab, size=base_length))\
                          + '/'.join(np.random.choice(vocab, size=path_length))

def gen_gYear(min_year=1970, max_year=2100):
    return np.random.randint(min_year, max_year)

def gen_date(min_year=1970, max_year=2100):
    start_date = datetime.date(1970, 1, 1)
    end_date = datetime.date(2100, 12, 31)
    num_days = (end_date - start_date).days
    rand_days = random.randint(1, num_days)
    random_date = str(start_date + datetime.timedelta(days=rand_days))
    return random_date

def gen_dateTime(min_time='00:00:00', max_time='23:59:59'):
    min_time = min_time.split(':')
    max_time = max_time.split(':')
    return '%sT%02d:%02d:%02d' % (gen_date(),
                            np.random.randint(int(min_time[0]), int(max_time[0])+1),
                            np.random.randint(int(min_time[1]), int(max_time[1])+1),
                            np.random.randint(int(min_time[2]), int(max_time[2])+1))

def gen_integer(min_value=-9e5, max_value=9e5):
    return np.random.randint(min_value, max_value)

def gen_float(min_value=-9e5, max_value=9e5):
    return np.random.rand() * np.random.randint(min_value, max_value)

def gen_boolean():
    return np.random.rand() > 0.5

def gen_image(size=(200, 200)):
    width = max(np.random.randint(min(size[0], size[1])//2), 1)
    x0 = np.random.randint(0, size[0])
    x1 = np.random.randint(0, size[0])
    y0 = np.random.randint(0, size[1])
    y1 = np.random.randint(0, size[1])
    im = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(im)
    draw.line([(x0, y0), (x1, y1)], fill="white", width=width)

    return im

def gen_point(min_lat=-90, max_lat=90, min_lon=-180, max_lon=180):
    return (gen_float(min_lon, max_lon), gen_float(min_lat, max_lat))

def gen_wktLiteral(min_length=4, max_length=16):
    num_points = np.random.randint(min_length, max_length)
    points = [gen_point() for _ in range(num_points-1)]
    points_str = ["{} {}".format(lon, lat) for (lon, lat) in points]
    points_str.append(points_str[0])  # close polygon
    return "POLYGON ((" +\
            ", ".join(points_str) +\
            "))"