#!/usr/bin/env sage

from collections import Counter
from random import randint
from time import time


def subdivide(G):
    g = G.copy()

    current = None
    mult = 1

    added = list()

    for edge in g.multiple_edges(labels=False, sort=True):
        if not current:
            current = edge
            continue

        if current != edge:
            a, b = current

            for i in range(mult):
                v = g.add_vertex()
                g.add_edges([(a, v), (v, b)])
                g.delete_edge(current)
                added.append(v)

            current = edge
            mult = 1
            continue

        mult += 1

    if current:
        a, b = current

        for i in range(mult):
            v = g.add_vertex()
            g.add_edges([(a, v), (v, b)])
            g.delete_edge(current)
            added.append(v)

    g.allow_multiple_edges(False)
    return g, added


def gomory_hu(g, algorithm=None, added=None):
    # Compute the subdivided graph
    # which has multiedged set to False already
    if added is None:
        gs, added = subdivide(g)

        # Now we can compyte the Gomory-Hu tree
        # via teh native Sage method
        ght = gs.gomory_hu_tree()
    else:
        ght = g.gomory_hu_tree()

    # For the sake of simplicity we delete the
    # additional verices that were added during
    # the subdividing procedure
    for v in added:
        neigs = ght.neighbors(v)
        neigs_len = len(neigs)

        # Simple case
        if neigs_len <= 1:
            ght.delete_vertex(v)
            continue

        # We need to generate a path of te neighbors of
        # the current added  vertex v
        to_add = []
        for i in range(neigs_len-1):
            x, y = neigs[i], neigs[i+1]
            l = min(ght.edge_label(v, w) for w in (x, y))
            to_add.append((x, y, l))

        ght.add_edges(to_add)
        ght.delete_vertex(v)

    return ght


def ght_local_conn(gh_tree, u, v):
    """
    Computes the Gomory-Hu tree min. label
    in the path between u and v.
    """
    path = gh_tree.shortest_path(u, v, by_weight=True)

    if len(path) == 0:
        return 0

    # Get the minimum label of the path
    return min(
        gh_tree.edge_label(path[i], path[i+1])
        for i in range(len(path)-1)
    )


def local_edge_connectivity(G, indicator=None):
    """
    Computes the local edge connectivity
    of G using the indicator vertex given.
    """

    t = gomory_hu(G)

    if indicator is None:
        indicator = t.vertices()[0]

    return min(
        ght_local_conn(t, indicator, other)
        for other in t
        if other != indicator
    )


def max_violation(g, ght, x, indicator, req):
    """
    Computes the max. violation of the edge connectivity
    requirement in the graph g.
    """
    return max(
        req - ght_local_conn(ght, w, indicator)
        for w in g.vertex_iterator()
        if w != x and w != indicator
    )


def split_off_attempt(g, x, indicator, req, candidates=None, verbose=False):
    """
    Attempt to split-off to capacity the vertex x
    with the candidates given.

    Returns
    -------
        The capacity of the splitting off
    """
    if candidates is None:
        neighbors = g.neighbor_iterator(x)
        u, v = next(neighbors), next(neighbors)
    else:
        u, v = candidates

    # Compute the minimum degree to which we will
    # split-off to capacity
    cap = min(
        len(g.edge_boundary([x], [u])),
        len(g.edge_boundary([x], [v]))
    )

    if u == v:
        cap //= 2

    if verbose:
        print("Boundary[u]: ", x, u, g.edge_boundary([x], [u]))
        print("Boundary[v]: ", x, v, g.edge_boundary([x], [v]))

    # Create H from G and make splitting
    h = g.copy()

    # Add new edges only if not loops
    if u != v:
        to_add = [(u, v) for _ in range(cap)]
        h.add_edges(to_add)

    # Delete old edges
    to_delete = [e for _ in range(cap) for e in [(x, u), (x, v)]]
    h.delete_edges(to_delete)

    # Compute the maximum violation of the
    # edge connectivity requirement
    ght = gomory_hu(h)

    if verbose:
        print(f"split_off_attempt POST")
        h.plot(layout='circular').show()
        print(h.edges())
        ght.plot(edge_labels=True).show()

    m_conn = min(
        ght_local_conn(ght, indicator, w)
        for w in ght
        if w != x and w != indicator
    )

    q = 0 if m_conn > req else req - m_conn

    if verbose:
        print(f"split-off: ({u}, {v}) to cap = {cap} and viol = {q} and req = {req}")

    # Re-compute the cap
    from math import ceil
    cap -= ceil(float(q) / 2.)

    return cap if cap > 0 else 0


def get_indicator(G, no):
    """
    Get a random vertex that is not in the
    iterable no.
    Used to get a clean indicator vertex in
    the splitting-off algorithm.
    """
    for v in G:
        if v not in no:
            return v

    return None


def split_off(G, x, req, iter_max=1000, verbose=False):
    """
    Computes the complete splitting-off sequence of x in G
    """

    H = G.copy()

    # Pick an indicator vertex.
    # We know that, in our case, we can pick any vertex as
    # an indicator vertes. So we will pick it in the process
    # Proof in notes.

    # Make the greedy split-off attempts, i.e.
    # apply Lemma 3.2 to find the splitting-off sequence
    # in O(|N|) time where N = |neighbors(x)|

    C = set()

    # Here we will save the splitting-off sequence
    added, removed = [], []

    i = 1
    while True:
        # Compute the set N(x) - C
        candidates = set(H.neighbor_iterator(x)).difference(C)
        if not candidates:
            break

        if verbose:
            print(f"Split({i}): {C} | {candidates}")

            if i > 2:
                H.show()

        i += 1

        # Pick any candidate vertex
        u = candidates.pop()

        if i > iter_max:
            raise Exception(f"Exceeded max iterations ({i})")

        # Case when |C| = 0
        # We simply assign C = {u} and continue
        if not C:
            C.add(u)
            continue

        # Trivial case when |C| = 1
        if len(C) == 1:
            # Pick v in C and split off (u, v) to capacity
            v = C.pop()

            # Pick a clean indicator
            indicator = get_indicator(H, [x])

            # Try to split-off to capacity (xu, xv)
            cap = split_off_attempt(H, x, indicator, req, candidates=(u, v), verbose=verbose)

            # Make the real splitting-off
            # if there's margin
            if cap > 0:
                # Delete old edges
                to_delete = [(x, u) for _ in range(cap)] + [(x, v) for _ in range(cap)]
                H.delete_edges(to_delete)

                # Add new edges
                to_add = [(u, v) for _ in range(cap)]
                H.add_edges(to_add)

                # Update the general splits list
                added += to_add
                removed += to_delete

            # Case (1)
            # Check if we voided any of u or v
            # if not (2) add them to C as non-admissible
            # i.e. only add u and v to C if they still have
            # neighbors

            if len(H.edge_boundary([x], [u])) != 0:
                 C.add(u)

            if len(H.edge_boundary([x], [v])) != 0:
                 C.add(v)

            # Go to the following iteration
            continue

        # General case, i.e. |C| > 1

        # Get to verrtices from the non-admissible set C
        v1, v2 = C.pop(), C.pop()
        if verbose:
            print(f"Split({i-1}): (u, v1, v2) = ({u}, {v1}, {v2})")

        # Pick a clean indicator
        indicator = get_indicator(H, [x])
        if verbose:
            print(f"Split({i-1}): indicator = {indicator}")

        # Split-off to capacity (u, v1)
        cap1 = split_off_attempt(H, x, indicator, req, candidates=(u, v1), verbose=verbose)

        if verbose:
            print(f"Split({i-1}): cap1 = {cap1}")

        to_delete = [(x, u) for _ in range(cap1)] + [(x, v1) for _ in range(cap1)]
        H.delete_edges(to_delete)

        to_add = [(u, v1) for _ in range(cap1)]
        H.add_edges(to_add)

        if cap1 > 0:
            added += to_add
            removed += to_delete

        # Split-off to capacity (u, v2)
        cap2 = split_off_attempt(H, x, indicator, req, candidates=(u, v2), verbose=verbose)

        if verbose:
            print(f"Split({i-1}): cap2 = {cap2}")

        to_delete = [(x, u) for _ in range(cap2)] + [(x, v2) for _ in range(cap2)]
        H.delete_edges(to_delete)

        to_add = [(u, v2) for _ in range(cap2)]
        H.add_edges(to_add)

        if cap2 > 0:
            added += to_add
            removed += to_delete

        # Check if any of the neighbors (u, v1, v2) were voided
        # in the process, if so we go further
        # to the following iteration.
        case_1 = False

        if len(H.edge_boundary([x], [u])) == 0:
            case_1 = True
        else:
            C.add(u)

        if len(H.edge_boundary([x], [v1])) == 0:
            case_1 = True
        else:
            C.add(v1)

        if len(H.edge_boundary([x], [v2])) == 0:
            case_1 = True
        else:
            C.add(v2)

        if case_1:
            continue

        # IMPORTANT
        # We need to rememeber that we may already have added
        # the set of neighbors (u, v1, v2) to C

        # If we arrive to this point, then we have a situation where
        # none of the x-neighbors voided.
        #
        # Now we check if v1 and v2 are contained in a tight set
        # by making a single splitting-off attempt of (xv1, xv2)

        # Make a single split-off
        # and check the connectivity after it
        h = copy(H)
        h.add_edge((v1, v2))
        h.delete_edges([(x, v1), (x, v2)])

        # If there was a connectivity violation
        # we assume that situation (2) applies, i.e.
        # that (v1, v2) are contained in a tight set
        if local_edge_connectivity(h, indicator) < req:
            continue

        # If everything went ok then situation (3) applies.

        # Make the real split-off
        H.add_edge((v1, v2))
        H.delete_edges([(x, v1), (x, v2)])

        # Remove v1 and v2 from C
        C.remove(v1)
        C.remove(v2)

        added += [(v1, v2)]
        removed += [(x, v1), (x, v2)]

    if len(H.neighbors(x)) != 0:
        if verbose:
            H.show()
        raise Exception(f"Could not split-off {x}")

    return C, added, removed


def lovasz_simplification(G, req, verbose=False):
    add, rm, rm_v = [], [], []

    step = 0
    while len(G) > 2:
        # Remove unneded edges (minimally edge connected)
        edges = G.edges()

        for e in edges:
            G.delete_edge(e)
            if local_edge_connectivity(G) < req:
                G.add_edge(e)
                continue

            add.insert(0, [])
            rm.insert(0, [e])
            rm_v.insert(0, None)

        if verbose:
            print("PRE")
            G.plot(layout='circular').show()

        # Pick the vertex to split-off
        x = None
        for v in G.vertex_iterator():
            if G.degree(v) == req:
                x = v
                break

        if x is None:
            raise Exception(f"Could not find vertex of degree {req}")

        if verbose:
            print(f"#{step:d}: splitting-off on ({x})")

        # Split-off every (u, u) to capacity
        # i.e., removes as many admissible pairs of (xu, xu) as possible
        should_continue = True
        for u in G.neighbor_iterator(x):
            indicator = get_indicator(G, [x, u])
            cap = split_off_attempt(G, x, indicator, req, candidates=(u, u), verbose=verbose)

            if cap > 0:
                if verbose:
                    print(f"pre split-off ({x}-{u}, {x}-{u}) to cap = {cap} ind = {indicator}")
                    G.plot(layout='circular').show()

                # Remove the pertinent edges
                to_delete = [(x, u) for _ in range(cap)] + [(x, u) for _ in range(cap)]
                G.delete_edges(to_delete)

                # Generate added list but not actually add the loops
                # as they would be removed
                to_add = [(u, u) for _ in range(cap)]

                add.insert(0, to_add)
                rm.insert(0, to_delete)

                # Check if we actually voided x
                if len(G.neighbors(x)) == 0:
                    G.delete_vertex(x)
                    rm_v.insert(0, x)
                    should_continue = False
                    break
                else:
                    rm_v.insert(0, None)

                if verbose:
                    G.plot(layout='circular').show()

        if not should_continue:
            step +=1
            continue

        # Try to split-off to capacity x
        # using the Lemma 3.2 from efficient edge split.
        C, added, removed = split_off(G, x, req, verbose=verbose)

        if len(C) > 0:
            raise Exception(f"Got not empty non-admissible set C: {C}")

        # Handle the delete
        G.add_edges(added)
        G.delete_edges(removed)
        G.delete_vertex(x)

        if verbose:
            print("POST", step, x, added, removed)
            G.plot(layout='circular').show()

        add.insert(0, added)
        rm.insert(0, removed)
        rm_v.insert(0, x)

        step += 1

    # Final cleanup
    while True:
        e = G.random_edge()

        G.delete_edge(e)
        if local_edge_connectivity(G) < req:
            G.add_edge(e)
            break

        add.insert(0, [])
        rm.insert(0, [e])
        rm_v.insert(0, None)

    return G, (add, rm, rm_v)


def random_orientation(G):
    G.to_directed()

    D = DiGraph(data=[G.vertices(), []],
                format='vertices_and_edges',
                multiedges=G.allows_multiple_edges(),
                loops=G.allows_loops(),
                weighted=G.weighted(),
                pos=G.get_pos(),
                name="Random orientation (v2) of {}".format(G.name()))

    if hasattr(G, '_embedding'):
        D._embedding = copy(G._embedding)

    edges = G.edges()

    from random import shuffle
    looper = [i for i in range(len(edges))]
    shuffle(looper)

    for i in looper:
        u, v, l = edges[i][0], edges[i][1], edges[i][2]

        if i % 2:
            D.add_edge(u, v, l)
        else:
            D.add_edge(v, u, l)

    return D


def lovasz_orientation(G, k, verbose=False):
    if not G.allows_multiple_edges():
        G.allow_multiple_edges(True)

    # Simplify via Lovasz theorem
    g, ops = lovasz_simplification(G, k, verbose=verbose)

    # Generate random orientation
    g = random_orientation(g)

    # Go backwards via Lovasz theorem
    for added, removed, v in zip(*ops):
        if v is not None:
            g.add_vertex(v)
        else:
            g.add_edges(removed)
            continue

        for s, e in added:
            edge = (s, e)

            if not g.has_edge(s, e):
                edge = (e, s)

            g.delete_edge(edge)
            g.add_edges([(edge[0], v), (v, edge[1])])

    return g


def EnOPODS(D, F, verbose=False):
    """
    Parameters
    ----------
        D is a digraph (V, A)
        F is a subset of edges if A
    """
    if Counter(D.edges()) != Counter(F):
        a = None

        # Pick a random edge from A
        for e in D.edge_iterator():
            if e not in F:
                a = e
                break

        if verbose:
            print(f"EnOPODS: F = {F}, a = {a}")

        u, v, _ = a

        # Call EnOPODS with the edge a in F
        yield from EnOPODS(D.copy(), F + [a], verbose=verbose)

        g = D.copy()
        g.delete_edges(F)

        path = g.shortest_path(v, u)
        if len(path) > 1:
            # Reverse path
            for i in range(len(path)-1):
                D.delete_edge(path[i], path[i+1])
                D.add_edge(path[i+1], path[i])

            D.delete_edge(a[0], a[1])
            D.add_edge(a[1], a[0])

            yield from EnOPODS(D.copy(), F + [(a[1], a[0], a[2])], verbose=verbose)

    else:
        if verbose:
            print(f"EnOPODS: yielding!")
        yield D.copy()


def out_degree_sequence(D):
    out_degree = []

    for i, v in enumerate(D):
        out_degree.append(0)

        for u in D.neighbor_iterator(v):
            if D.has_edge(v, u):
                out_degree[i] += 1

    return out_degree


def eo_algo_2(G, req, verbose=False):
    D = lovasz_orientation(G, req)
    if verbose:
        g = D.copy()
        g.allow_multiple_edges(False)
        assert g.is_isomorphic(D)
        print(f"eo_algo_2: edge conn. of D is {g.edge_connectivity()}")
    return EnOPODS(D.copy(), list(), verbose=verbose), out_degree_sequence(D)


def _bfs(G, u, v):
    bfs = list(G.breadth_first_search(u, edges=True))

    path = []
    parent = v

    # Backtrack path generation
    for i in range(len(bfs)-1, -1, -1):
        x, y = bfs[i]
        if y == parent:
            parent = x
            path.append(y)

    path.append(parent)
    return path


def _reverse(G, u, v):
    path = _bfs(G, u, v)

    # Reverse path and re-iterate
    for i in range(len(path)-1):
        x, y = path[i], path[i+1]
        G.delete_edge(y, x)
        G.add_edge((x, y))

    return path


def is_flippable(D, u, v, req=None, step=0):
    """
    O(km) algorithm to check if the path from u to v is flippable
    """

    # If we applied this procedure req + 1 times
    # then the original (u, v) is flippable
    # i.e. at most we will recurse req = k times
    if step > req:
        return True

    # O(m) complexity BFS path search
    path = _bfs(D, u, v)

    # Check if there still exists the path from u to v
    if len(path) < 2:
        return False

    # Reverse path and re-iterate
    for i in range(len(path)-1):
        x, y = path[i], path[i+1]
        D.delete_edge(y, x)
        D.add_edge((x, y))

    # Recursive step
    return is_flippable(D, u, v, req=req, step=step+1)


def EnODS(D, F, req=None, verbose=verbose, orientation=False):
    if verbose:
        print(f"EnODS: F = {F}")
    if Counter(D) != Counter(F):
        v = None
        for w in D:
            if w not in F:
                v = w
                break

        yield from reverse_neg(D.copy(), F, v, req=req, verbose=verbose)
        yield from reverse_pos(D.copy(), F, v, req=req, verbose=verbose)
        yield from EnODS(D.copy(), F + [v], req=req, verbose=verbose)
    else:
        if verbose:
            print(f"EnODS: yielding! ({out_degree_sequence(D)})")

        if orientation:
            yield out_degree_sequence(D)
        else:
            yield D.copy()


def reverse_neg(D, F, v, req=None, seen=[], verbose=verbose, algorithm=EnODS):
    if verbose:
        print(f"Reverse-: F = {F} and v = {v} and req = {req} and seen = {seen}")

    candidate = None
    for u in D:
        if u == v:
            continue

        if u not in F and u and is_flippable(D.copy(), v, u, req=req):
            candidate = u
            break

    if candidate is not None:
        path = _reverse(D, v, candidate)
        if verbose:
            print(f"Reverse- ({candidate}) path: {path}")

        yield from reverse_neg(D.copy(), F, v, req=req, seen=seen+[candidate], verbose=verbose, algorithm=algorithm)
        yield from algorithm(D.copy(), F + [v], req=req, verbose=verbose)
    else:
        if verbose:
            print(f"Reverse-: candidate is None")
        pass


def reverse_pos(D, F, v, req=None, seen=[], verbose=verbose, algorithm=EnODS):
    if verbose:
        print(f"Reverse+: F = {F} and v = {v} and req = {req} and seen = {seen}")

    candidate = None
    for u in D:
        if u == v:
            continue

        if u not in F and u and is_flippable(D.copy(), u, v, req=req):
            candidate = u
            break

    if candidate is not None:
        path = _reverse(D, candidate, v)
        if verbose:
            print(f"Reverse+ ({candidate}) path: {path}")

        yield from reverse_pos(D.copy(), F, v, req=req, seen=seen+[candidate], verbose=verbose, algorithm=algorithm)
        yield from algorithm(D.copy(), F + [v], req=req, verbose=verbose)
    else:
        if verbose:
            print(f"Reverse+: candidate is None")
        pass


def EnODS2(D, F, req=None, verbose=False):
    if Counter(D) != Counter(F):
        # Pick the smallest v in V \ F
        # as there's no linear ordering
        # we pick any
        v = None
        for w in D:
            if w not in F:
                v = w
                break

        if verbose:
            print(f"EnODS2: enter with F = {F} and v = {v}")


        yield from reverse_pos(D.copy(), F, v, req=req, verbose=verbose, algorithm=EnODS2)
        yield from reverse_neg(D.copy(), F, v, req=req, verbose=verbose, algorithm=EnODS2)
        yield from EnODS2(D.copy(), F + [v], req=req, verbose=verbose)
    else:
        # Case then we want all
        if verbose:
            print(f"EnODS2: yielding from EnOPODS with {out_degree_sequence(D)}")
        yield from EnOPODS(D.copy(), [], verbose=verbose)


def eo_algo_3(G, req, verbose=False):
    D = lovasz_orientation(G.copy(), req, verbose=False)
    return EnODS(D.copy(), [], req=req//2, verbose=verbose, orientation=True)


def eo_algo_4(G, req, verbose=False):
    D = lovasz_orientation(G.copy(), req, verbose=False)
    return EnODS2(D.copy(), [], req=req//2, verbose=verbose)
