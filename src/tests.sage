# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: SageMath 9.1
#     language: sage
#     name: sagemath
# ---

# + active=""
# print('example-test')
# start = time()
#
# try:
#     pass
# except Exception as e:
#     print(f"FAIL {e}")
#     raise e
#     exit(1)
#     
# print(f'OK {time()-start:4.4f} s.')

# +
from time import time

load(f'lib.sage')

# +
print(f'subdivide')
start = time()

try:
    # Given a multi-graph with two vertices
    # and n edges between them, the subdivided graph
    # should have 2 + n vertices 2 * n edges.
    # Also we should be able to make the graph simple
    
    for n in range(2, 11):
        u, v = 0, 1
        edges = [(u, v) for _ in range(n)]
        
        g = Graph({u: [v for _ in range(n)]})
        
        assert len(g) == 2
        assert len(g.edges()) == n
        
        new_g, added = subdivide(g)
        
        assert len(new_g) == 2 + n
        assert len(new_g.edges()) == 2 * n
        
        assert len(added) == n
        assert added == [i for i in range(2, n+2)]
        
        new_g.allow_multiple_edges(True)
        
        # Check that for all added vertices v_i we have the
        # wanted edges {u, v_i}, {v_i, v}
        for added_v in added:
            assert new_g.has_edge(u, added_v) and new_g.has_edge(added_v, v)
except Exception as e:
    print(f"FAIL {e}")
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('gomory_hu_tree')
start = time()

try:
    # First of all, for a simple graph we should be able to
    # call gomory_hu and get the same resulting tree
    
    for n in range(3, 10):
        g = graphs.CompleteGraph(n)
        assert gomory_hu_tree(g) == g.gomory_hu_tree()
    
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
            
    t = gomory_hu_tree(g)
    new_k = min(w for _, _, w in t.edges())
    
    assert new_k == k + 1
except Exception as e:
    print(f"FAIL {e}")
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('connectivity')
start = time()

try:
    from itertools import combinations
    
    g = graphs.CompleteGraph(7)
    t = g.gomory_hu_tree()
    all_pairs = combinations(g.vertices(), 2)
    assert all(local_connectivity(t, u, v) == 6 for u, v in all_pairs)

    # For a K(n) we have that
    # \lambda(u, v) = n - 1, for all u, v
    n = 7
    g = graphs.CompleteGraph(n)
    
    for u in g:
        for v in g:
            if u == v:
                continue
            
            assert len(g.edge_disjoint_paths(u, v)) == n - 1
    
    g.allow_multiple_edges(True)
    
    for u, v, _ in g.edges()[:n]:
        g.add_edge(u, v, None)
            
    t = gomory_hu_tree(g)
    
    conn = None
    
    # We now check that for all pairs {u, v}
    # the local edge connectivity is at least n
    for u in t:
        for v in t:
            if u == v:
                continue
                
            current = local_connectivity(t, u, v)
            conn = current if conn is None or current <= conn else conn
            
            assert current >= n
            
    # Check connectivity of the whole graph
    # via Gomory-Hu tree returns correct results.
    # 
    # We will use the NetworkX function to check it
    # as it supports multi-graphs.
    from networkx import edge_connectivity
    
    got = connectivity(g)
    want = edge_connectivity(g.networkx_graph())
    
    assert conn == got and got == want
except Exception as e:
    print(f"FAIL {e}")
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('splitting-off')
start = time()

try:
    n = 7
    # We want that for a given req the graph
    # after splitting to capacity the pair {2, 3}
    # satisfies the req.
    g = Graph({1: [2, 2, 2, 2, 3, 3, 3], 4: [2, 2, 3, 3]})
    
    # The initial connectivity of g will
    # be the requirement, i.e. its global.
    req = connectivity(g)
    
    # We split-off 1 and pick 4 as indicator
    x, indicator = 1, 4
    
    # Here we split-off to capcacity
    cap = splitting_off_to_capacity(g, x, indicator, req, candidates=(2, 3))
    for _ in range(cap):
        g.delete_edges([(1, 2), (1, 3)])
        g.add_edge(2, 3)
    
    
    # Compute the minimum local edge-connectivity
    # of the resulting graph via all vertex pairs
    # that do not contain x
    t = gomory_hu_tree(g)
    after_conn = None
    
    for u in g:
        for v in g:
            if u == v or x in (u, v):
                continue
                
            l_conn = local_connectivity(t, u, v)
            
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
    
    complete_splitting_off(g, x, req)
    
    # Check that the edge connectivity after the
    # complete splitting-off is preserved.
    after_conn = connectivity(g)
    assert req == after_conn
except Exception as e:
    print(f"FAIL {e}")
    raise e
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('orientation')
start = time()

try:
    # We know that a K_n is (n-1) connected
    # hence the Lovasz decomposition of a K_n
    # with odd degree such that (n-1) >= 2 should yield
    # a (n-1) / 2 connected orientation.
    for n in (3, 5, 7):
        g = graphs.CompleteGraph(n)
        g.allow_multiple_edges(True)
        
        req = n - 1
        k = req // 2
        
        # By the Lovasz decomposition Theorem
        # h should have 2 vertices and 2k edges
        # connecting those vertices.
        h, _ = lovasz_decomposition(copy(g), req, verbose=False)
        assert len(h) == 2 and len(h.edges()) == 2 * k
        
        # Now let's actually generate a k-connected
        # orientation of g
        ori = orientation(copy(g), k)
        
        # The following properties should be true
        #  1. the undirected version of ori should be g
        #  2. ori should be k-connected
        assert ori.to_undirected() == g
        
        # An orientation of K_n should not have 
        # multiple edges
        assert ori.has_multiple_edges() == False
        ori.allow_multiple_edges(False)
        
        assert ori.edge_connectivity() == k
    
    for n in (4, 6, 8):
        # When n is even it should raise an exception
        # as the splitting-off should not work
        
        g = graphs.CompleteGraph(n)
        g.allow_multiple_edges(True)
        
        req = n - 1
        
        failed = False
        try:
            lovasz_decomposition(copy(g), req)
        except:
            failed = True
            
        if not failed:
            raise Exception(f'"lovasz_decomposition" should have failed for K_{n}')
except Exception as e:
    print(f"FAIL {e}")
    raise e
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('enumeration-helpers')
start = time()

try:
    g = DiGraph({
        1: [2],
        2: [3, 4],
    })
    
    # _bfs should return correct reversed
    # path from 1 to 4: 4 -> 2 -> 1
    path = _bfs(g, 1, 4)
    assert path == [4, 2, 1]
    
    # _reverse should return correct reversed
    # path from 1 to 4: 4 -> 2 -> 1 and
    # reverse it in g
    path2 = _reverse(g, 1, 4)
    assert path == path2
    
    # check that the path was actually reversed in
    # the original graph
    for u, v in ((1, 2), (2, 4)):
        assert not g.has_edge(u, v) and g.has_edge(v, u)
        
    # Compute a 3-connected orientation of K_7
    d = orientation(graphs.CompleteGraph(7), 3)
    d.allow_multiple_edges(False)
    
    assert d.edge_connectivity() == 3
    
    from itertools import combinations
    
    # As d is 3-connected, we should be able to flip every
    # vertex pair for k = 1 (resp. 2), as it means we can 
    # find 2 (resp. 3) edge-disjoint paths for every pair.
    # Also, we should not be able to flip any pair with k = 3.
    assert all(_is_flippable(d.copy(), u, v, 1) for u, v in combinations(d.vertices(), 2))
    assert all(_is_flippable(d.copy(), u, v, 2) for u, v in combinations(d.vertices(), 2))
    assert all(not _is_flippable(d.copy(), u, v, 3) for u, v in combinations(d.vertices(), 2))
except Exception as e:
    print(f"FAIL {e}")
    raise e
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('enumeration')
start = time()

try:
    from sage.graphs.orientations import strong_orientations_iterator
    
    # For k = 1 we should get the same result as the
    # native strong orientations iterator from Sage
    for n in (3, 5):
        g = graphs.CompleteGraph(n)
        
        new = len(list(k_orientations_iterator(g, 1)))
        strong = len(list(strong_orientations_iterator(g))) * 2
        
        assert new == strong
        
    query = GraphQuery(
        display_cols=['graph_id', 'num_vertices', 'edge_connectivity'], 
        edge_connectivity=['=', 4], 
        num_vertices=['<', 7], 
        num_edges=['<', 15], 
    )
    
    items = query.get_graphs_list()
    max_items = 3
    
    from random import choices

    for item in choices(items, k=max_items):
        g = item.copy()
        g.allow_multiple_edges(True)
        
        got_1, got_2 = 0, 0
        for ori in strong_orientations_iterator(g):
            got_1 += 1
            
            if ori.edge_connectivity() == 2:
                got_2 += 1
                
        got_1 *= 2
        got_2 *= 2
        
        want_1 = len(list(k_orientations_iterator(g.copy(), 1)))
        want_2 = len(list(k_orientations_iterator(g.copy(), 2)))
        
        assert got_1 == want_1 and got_2 == want_2
except Exception as e:
    print(f"FAIL {e}")
    raise e
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')

# +
print('enumeration-results')
start = time()

try:
    # We know that this graph has 3842 2-connected orientations 
    # Conclusions arXiv:1908.02050
    g1 = {
        1: [2, 3, 8, 9],
        2: [3, 4, 8],
        3: [4, 9],
        4: [5, 6, 8, 9],
        5: [6, 7, 8],
        6: [7, 9],
        7: [8, 9],
        8: [9],
    }
    
    assert len(list(k_orientations_iterator(Graph(g1), 2))) == 3842
except Exception as e:
    print(f"FAIL {e}")
    raise e
    exit(1)
    
print(f'OK {time()-start:4.4f} s.')
# -


