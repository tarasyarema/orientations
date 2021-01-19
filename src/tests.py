#!/usr/bin/env python
# coding: utf-8

# - `subdivide`
# - `gomory_hu` should be `gomory_hu_tree`
# - `ght_local_conn` should be `local_connectivity`
# - `local_edge_connectivity` should be `connectivity`
# - `max_violation` should be `max_connectivity_violation`
# - `split_off_attempt` should be `splitting_off_to_capacity`
# - `get_indicator`
# - `split_off` should be `complete_splitting_off`
# 
# #### TODO
# 
# - `lovasz_simplification` should be `lovasz_decomposition`
# - `random_orientation`
# - `lovasz_orientation` should be `orientation`
# - `EnOPODS`
# - `out_degree_sequence`
# - `eo_algo_2` can be removed
# - `_bfs`
# - `_reverse`
# - `is_flippable`
# - `EnODS` can be removed
# - `reverse_neg`
# - `reverse_pos`
# - `EnODS2`
# - `eo_algo_3` can be removed
# - `eo_algo_4` should be `k_orientations_iterator`

# In[1]:


from sys import exit
from pathlib import Path

# Import the actual lib
load(f'{Path.cwd()}/lib.sage')


# ### `subdivide` test

# In[2]:


try:
    # Given a multi-graph with two vertices
    # and n edges between them, the subdivided graph
    # should have 2 + n vertices 2 * n edges.
    # Also we should be able to make the graph simple
    
    for n in range(2, 11):
        u, v = 1, 2
        edges = [(u, v) for _ in range(n)]
        
        g = Graph({u: [v for _ in range(n)]})
        
        assert len(g) == 2
        assert len(g.edges()) == n
        
        new_g, added = subdivide(g)
        
        assert len(new_g) == 2 + n
        assert len(new_g.edges()) == 2 * n
        
        new_g.allow_multiple_edges(True)
        
        # Check that for all added vertices v_i we have the
        # wanted edges {u, v_i}, {v_i, v}
        for added_v in added:
            assert new_g.has_edge(u, added_v) and new_g.has_edge(added_v, v)
except Exception as e:
    print(f"subdivide: FAIL {e}")
    exit(1)
    
print("subdivide: OK")


# ### `gomory_hu` tests

# In[3]:


try:
    # First of all, for a simple graph we should be able to
    # call gomory_hu and get the same resulting tree
    
    for n in range(3, 10):
        g = graphs.CompleteGraph(n)
        assert gomory_hu(g) == g.gomory_hu_tree()
    
    # If the given graph is multiedged it should
    # handle this correctly and keep edge-connectivity
    n = 5
    g = graphs.CompleteGraph(n)
    
    k = g.edge_connectivity()
    g.allow_multiple_edges(True)
    
    # We add a n = 5 new edges we should
    # making the \lambda_G'(u, v) = \lambda_G(u, v) + 1
    for u, v, _ in g.edges()[:n]:
        g.add_edge(u, v, None)
            
    t = gomory_hu(g)
    new_k = min(w for _, _, w in t.edges())
    
    assert new_k == k + 1
except Exception as e:
    print(f"gomory_hu: FAIL {e}")
    exit(1)
    
print("gomory_hu: OK")


# ### connectivity tests

# In[4]:


n = 7

try:
    # For a K(n) we have that
    # \lambda(u, v) = n - 1, for all u, v
    g = graphs.CompleteGraph(n)
    
    for u in g:
        for v in g:
            if u == v:
                continue
            
            assert len(g.edge_disjoint_paths(u, v)) == n - 1
    
    g.allow_multiple_edges(True)
    
    for u, v, _ in g.edges()[:n]:
        g.add_edge(u, v, None)
            
    t = gomory_hu(g)
    
    conn = None
    
    # We now check that for all pairs {u, v}
    # the local edge connectivity is at least n
    for u in t:
        for v in t:
            if u == v:
                continue
                
            current = ght_local_conn(t, u, v)
            conn = current if conn is None or current <= conn else conn
            
            assert current >= n
            
    # Check connectivity of the whole graph
    # via Gomory-Hu tree returns correct results.
    # 
    # We will use the NetworkX function to check it
    # as it supports multi-graphs.
    from networkx import edge_connectivity
    
    got = local_edge_connectivity(g)
    want = edge_connectivity(g.networkx_graph())
    
    assert conn == got and got == want
except Exception as e:
    print(f"connectivity: FAIL {e}")
    exit(1)
    
print("connectivity: OK")


# ### splitting-off tests

# In[5]:


n = 7

try:
    g = graphs.CompleteGraph(n)
    
    # We will remove one edge of the fisrt 
    # vertex and we will see that the
    # connectivity violation is 1
    before_conn = g.edge_connectivity()
    
    edge = g.edges()[0]
    g.delete_edge(edge)
    
    # We can pick any vertex as indicator
    indicator = g.vertices()[0]
    t = g.gomory_hu_tree()
    
    violation = max_violation(g, t, indicator, indicator, before_conn)
    assert violation == 1
    
    # Now we will check that the splitting-off
    # to capacity works.
    #
    # We want that for a given req the graph
    # after splitting to capacity the pair {2, 3}
    # satisfies the req.
    g = Graph({1: [2, 2, 2, 2, 3, 3, 3], 4: [2, 2, 3, 3]})
    
    # The initial connectivity of g will
    # be the requirement, i.e. its global.
    req = local_edge_connectivity(g)
    
    # We split-off 1 and pick 4 as indicator
    x, indicator = 1, 4
    
    # Here we split-off to capcacity
    cap = split_off_attempt(g, x, indicator, req, candidates=(2, 3))
    for _ in range(cap):
        g.delete_edges([(1, 2), (1, 3)])
        g.add_edge(2, 3)
    
    
    # Compute the minimum local edge-connectivity
    # of the resulting graph via all vertex pairs
    # that do not contain x
    t = gomory_hu(g)
    after_conn = None
    
    for u in g:
        for v in g:
            if u == v or x in (u, v):
                continue
                
            l_conn = ght_local_conn(t, u, v)
            
            if after_conn is None or l_conn <= after_conn:
                after_conn = l_conn
    
    assert req == l_conn
    
    # Now we finally test the complete splitting-off at x.
    g = graphs.CompleteGraph(n)
    
    # We know that, for even n, all vertices at g = K(n) have degree n - 1.
    # If we pick k = (n - 1) // 2, then we know that g has a vertex
    # of 2k and hence a complete splitting-off in it.
    req = g.edge_connectivity()
    g.allow_multiple_edges(True)
    
    # We may pick any x
    x = g.vertices()[0]
    
    split_off(g, x, req)
    
    # Check that the edge connectivity after the
    # complete splitting-off is preserved.
    after_conn = local_edge_connectivity(g)
    assert req == after_conn
except Exception as e:
    print(f"splitting-off: FAIL {e}")
    raise e
    exit(1)
    
print("splitting-off: OK")


# In[ ]:




