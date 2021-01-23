r"""
Enumerating `k`-connected orientations

This module is the code implemented in [TYT]_
based of the graph work on enumerating `k`-connected
orientations by [SKP]_.

References
----------

.. [SKP] Sarah Blind, Kolja Knauer, Petru Valicov
    *Enumerating k-arc-connected orientations*.
    https://arxiv.org/abs/1908.02050

.. [TYT] Taras Yarema
    *Enumerating k-connected orientations*.
    Degree thesis, University of Barcelona, 2021.


**This module contains the following main methods**

.. csv-table::
    :class: contentstable
    :delim: |

        :meth:`orientation` | Computes a `k`-connected orientation of `G`.
        :meth:`k_orientations_iterator` | Returns an iterator over all `k`-connected orientations of `G`.

**The following are helper functions that are exported**

.. csv-table::
    :class: contentstable
    :delim: |

        :meth:`subdivide` | Transform a multi-graph into a simple graph
        :meth:`gomory_hu_tree` | Gomory-Hu tree implementation with multi-graph support
        :meth:`local_connectivity` | Compute the min. edge label of a `u-v` path
        :meth:`connectivity` | Edge-connectivity of a undirected multi-graph `G`
        :meth:`splitting_off_to_capacity` | Split-off to capacity an edge pair
        :meth:`complete_splitting_off` | Efficient complete splitting-off at `x`
        :meth:`lovasz_decomposition` | Lovasz decomposition of `G`
        :meth:`alpha_orientations_iterator` | Iterator over all `\alpha`-orientations
        :meth:`outdegree_sequence_iterator` | Iterator over all `k`-connected outdegree sequences

Authors
-------

- Taras Yarema (2021-01-24) -- initial version

Methods
-------
"""


def subdivide(G):
    r"""
    Subdivides the multi-graph `G` such that the
    resulting graph preserves connectivity and it's
    not multi-edged.

    INPUT:

    - `G` -- a Graph that may have multiple edges.

    OUTPUT: `(H, C)` where `H` is the subdivided
    graph version of `G` and `C` is the list of
    added vertices.

    EXAMPLES:

    Simple usage for a graph formed by two vertices
    and a double edge between::

        sage: from orientations import subdivide
        sage: g = Graph({0: [1, 1]})
        sage: h, added = subdivide(g)
        sage: added
        [2, 3]
        sage: h.edges()
        [(0, 2, None), (0, 3, None), (1, 2, None), (1, 3, None)]

    TEST::

        sage: H, C = subdivide(graphs.CompleteGraph(5))
        sage: len(C)
        0
        sage: H == graphs.CompleteGraph(5)
        True

    Given a multi-graph with two vertices and `n`
    edges between them, the subdivided graph should
    have `2 + n` vertices `2n` edges. Also we
    should be able to make the graph simple::

        sage: for n in range(2, 11):
        ....:     u, v = 0, 1
        ....:     edges = [(u, v) for _ in range(n)]
        ....:     g = Graph({u: [v for _ in range(n)]})
        ....:     assert len(g) == 2
        ....:     assert len(g.edges()) == n
        ....:     new_g, added = subdivide(g)
        ....:     assert len(new_g) == 2 + n
        ....:     assert len(new_g.edges()) == 2 * n
        ....:     assert len(added) == n
        ....:     assert added == [i for i in range(2, n+2)]
        ....:     new_g.allow_multiple_edges(True)
        ....:     for added_v in added:
        ....:         assert new_g.has_edge(u, added_v)
        ....:         assert new_g.has_edge(added_v, v)
    """
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


def gomory_hu_tree(G, added=None):
    r"""
    Computes the Gomory-Hu tree representation
    of an undirected multi-graph G.

    INPUT:

    - `G` -- a Graph that may have multiple edges.

    OUTPUT: `T` a Gomory-Hu representation of `G`.

    EXAMPLES:

    Example usage::

        sage: from orientations import gomory_hu_tree
        sage: g = graphs.PetersenGraph()
        sage: g.allow_multiple_edges(True)
        sage: for edge in g.edges():
        ....:     g.add_edge(edge)
        sage: ght = gomory_hu_tree(g)
        sage: ght.is_tree()
        True
        sage: min(ght.edge_labels())
        6

    TEST:

    First of all, let's make sure that the implementation
    is consistent with the existing when the graph is simple::

        sage: for n in range(3, 10):
        ....:     g = graphs.CompleteGraph(n)
        ....:     assert gomory_hu_tree(g) == g.gomory_hu_tree()

    If the given graph is multiedged it should handle this
    correctly and keep edge-connectivity.
    We will then add 5 new edges and it should make the
    `\lambda_G'(u, v) = \lambda_G(u, v) + 1`::

        sage: g = graphs.CompleteGraph(5)
        sage: k = g.edge_connectivity()
        sage: g.allow_multiple_edges(True)
        sage: for u, v, _ in g.edges()[:5]:
        ....:     g.add_edge(u, v, None)
        sage: t = gomory_hu_tree(g)
        sage: new_k = min(w for _, _, w in t.edges())
        sage: assert new_k == k + 1
    """

    # Compute the subdivided graph
    # which has multiedged set to False already
    if added is None:
        gs, added = subdivide(G)

        # Now we can compyte the Gomory-Hu tree
        # via teh native Sage method
        ght = gs.gomory_hu_tree()
    else:
        ght = G.gomory_hu_tree()

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


def local_connectivity(T, u, v):
    r"""
    Computes the minimum label of the `u-v` path in `T`.
    As `T` is a Gomory-Hu tree representation of some `G`
    this label is the value of the `u-v` cut.

    INPUT:

    - `T` -- a Gomory-Hu tree.
    - `u` -- a vertex of `T`.
    - `v` -- a vertex of `T`.

    OUTPUT: The local edge-connectivity between
    `u` and `v`.

    EXAMPLES:

    Example usage::

        sage: from orientations import local_connectivity

    TEST::

        sage: g = graphs.CompleteGraph(7)
        sage: t = g.gomory_hu_tree()
        sage: all_pairs = Subsets(g.vertices(), 2)
        sage: assert all(local_connectivity(t, u, v) == 6 \
                for u, v in all_pairs)

    For a `K(n)` we have that `\lambda(u, v) = n - 1`,
    for all pairs `u, v`::

        sage: g = graphs.CompleteGraph(7)
        sage: for u in g:
        ....:     for v in g:
        ....:         if u == v:
        ....:             continue
        ....:         assert len(g.edge_disjoint_paths(u, v)) == 6

    We now check that for all pairs `{u, v}` the local
    edge-connectivity is at least `n`::

        sage: g = graphs.CompleteGraph(7)
        sage: g.allow_multiple_edges(True)
        sage: for u, v, _ in g.edges()[:7]:
        ....:     g.add_edge(u, v, None)
        sage: from orientations import gomory_hu_tree
        sage: t = gomory_hu_tree(g)
        sage: for u in t:
        ....:     for v in t:
        ....:         if u == v:
        ....:             continue
        ....:         current = local_connectivity(t, u, v)
        ....:         assert current >= 7
    """
    path = T.shortest_path(u, v, by_weight=True)

    if len(path) == 0:
        return 0

    # Get the minimum label of the path
    return min(
        T.edge_label(path[i], path[i+1])
        for i in range(len(path)-1)
    )


def connectivity(G, indicator=None):
    r"""
    Computes the local edge-connectivity of an
    undirected multi-graph `G`.

    INPUT:

    - `G` -- a Graph.

    OUTPUT: The edge-connectivity of `G`.

    EXAMPLES:

    We will mymic the Sage edge_connectivity examples.

    A basic application on the PappusGraph::

        sage: from orientations import connectivity
        sage: g = graphs.PappusGraph()
        sage: g.edge_connectivity()
        3
        sage: connectivity(g)
        3

    Even if obviously in any graph we know that the edge
    connectivity is less than the minimum degree of the graph::

        sage: g = graphs.RandomGNP(10,.3)
        sage: assert min(g.degree()) >= connectivity(g)

    TEST::

        sage: g = graphs.CompleteGraph(7)
        sage: assert connectivity(g) == 6
    """

    t = gomory_hu_tree(G)

    if indicator is None:
        indicator = t.vertices()[0]

    return min(
        local_connectivity(t, indicator, other)
        for other in t
        if other != indicator
    )


def splitting_off_to_capacity(g, x, indicator, req, candidates=None, verbose=False):
    r"""
    Attempt to do a splitting-off to capacity at the vertex `x`
    in `G`, such that it preserves the edge-connectivity requirements
    defined by req. The indicator is the a vertex of `G`. If the
    candidates are given the splitting-off is attempted with them.


    INPUT:

    - `G` -- a Graph.
    - `x` -- the vertex to split-off.
    - `indicator` -- a vertex.
    - `candidates` -- a pair of neighbors of `x`.
    - `verbose` -- if the users wants debugging output.

    OUTPUT: The capacity of the splitting off.

    EXAMPLES:

    Example usage::

        sage: from orientations import splitting_off_to_capacity

    TEST:

    The initial connectivity of g will be the requirement, i.e. its global.
    We split-off to capacity the vertex 1 and pick 4 as indicator.
    We could pick any vertex (non x) as indicator, as the requirement
    is global.
    We will then compute the minimum local edge-connectivity of the resulting
    graph via all vertex pairs that do not contain `x`::


        sage: from orientations import connectivity
        sage: g = Graph({1: [2, 2, 2, 2, 3, 3, 3], 4: [2, 2, 3, 3]})
        sage: req = connectivity(g)
        sage: x, indicator = 1, 4
        sage: cap = splitting_off_to_capacity(g, x, indicator, \
                req, candidates=(2, 3))
        sage: for _ in range(cap):
        ....:     g.delete_edges([(1, 2), (1, 3)])
        ....:     g.add_edge(2, 3)
        sage: from orientations import gomory_hu_tree
        sage: t = gomory_hu_tree(g)
        sage: after_conn = None
        sage: from orientations import local_connectivity
        sage: for u in g:
        ....:     for v in g:
        ....:         if u == v or x in (u, v):
        ....:             continue
        ....:         l_conn = local_connectivity(t, u, v)
        ....:         if after_conn is None or l_conn <= after_conn:
        ....:             after_conn = l_conn
        sage: assert req == after_conn
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
    ght = gomory_hu_tree(h)

    if verbose:
        print(f"split_off_attempt POST")
        h.plot(layout='circular').show()
        print(h.edges())
        ght.plot(edge_labels=True).show()

    m_conn = min(
        local_connectivity(ght, indicator, w)
        for w in ght
        if w != x and w != indicator
    )

    q = 0 if m_conn > req else req - m_conn

    if verbose:
        print(
            f"split-off: ({u}, {v}) to cap = {cap} and viol = {q} and req = {req}")

    # Re-compute the cap
    from math import ceil
    cap -= ceil(float(q) / 2.)

    return cap if cap > 0 else 0


def _get_indicator(G, no):
    r"""
    Gets a random vertex of `G` that is not in the
    iterable `no`. It is used to get a clean indicator
    vertex in the complete splitting-off algorithm.

    INPUT:

    - `G` -- a Graph.
    - `no` -- a iterable subset of the vertices of `G`.

    OUTPUTS: A vertex of `G` that is not in `no`.

    TEST::

        sage: g = graphs.CompleteGraph(5)
        sage: from orientations.orientations import _get_indicator
        sage: _get_indicator(g, [1, 2, 3, 4])
        0
    """

    for v in G:
        if v not in no:
            return v

    return None


def complete_splitting_off(G, x, req, verbose=False, iter_max=1000):
    r"""
    Computes the complete splitting-off sequence of `x` in `G`,
    such that it preserves the global edge-connectivity requirement.

    INPUT:

    - `G` -- a Graph.
    - `x` -- the vertex to split-off.
    - `req` -- the global edge-connectivity requirement.
    - `verbose` -- if the user wants debugging output.

    OUTPUT: `(C, A, R)` where `A` is a list with the added
    edges and `R` the removed ones.

    EXAMPLES:

    Example usage::

        sage: from orientations import complete_splitting_off

    TEST:

    We know that, for even n, all vertices at `g = K(n)` have
    degree `n - 1`. If we pick `k = (n - 1) // 2`, then we
    know that `g` has a vertex of degree `2k` and hence can
    compute a complete splitting-off in it::

        sage: g = graphs.CompleteGraph(5)
        sage: req = g.edge_connectivity()
        sage: g.allow_multiple_edges(True)
        sage: x = g.vertices()[0]
        sage: _ = complete_splitting_off(g, x, req)

    Finally, we check that the edge-connectivity requirement
    after the complete splitting-off is preserved::

        sage: from orientations import connectivity
        sage: after_conn = connectivity(g)
        sage: assert req == after_conn
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
            indicator = _get_indicator(H, [x])

            # Try to split-off to capacity (xu, xv)
            cap = splitting_off_to_capacity(
                H, x, indicator, req, candidates=(u, v), verbose=verbose)

            # Make the real splitting-off
            # if there's margin
            if cap > 0:
                # Delete old edges
                to_delete = [(x, u) for _ in range(cap)] + [(x, v)
                                                            for _ in range(cap)]
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
        indicator = _get_indicator(H, [x])
        if verbose:
            print(f"Split({i-1}): indicator = {indicator}")

        # Split-off to capacity (u, v1)
        cap1 = splitting_off_to_capacity(
            H, x, indicator, req, candidates=(u, v1), verbose=verbose)

        if verbose:
            print(f"Split({i-1}): cap1 = {cap1}")

        to_delete = [(x, u) for _ in range(cap1)] + [(x, v1)
                                                     for _ in range(cap1)]
        H.delete_edges(to_delete)

        to_add = [(u, v1) for _ in range(cap1)]
        H.add_edges(to_add)

        if cap1 > 0:
            added += to_add
            removed += to_delete

        # Split-off to capacity (u, v2)
        cap2 = splitting_off_to_capacity(
            H, x, indicator, req, candidates=(u, v2), verbose=verbose)

        if verbose:
            print(f"Split({i-1}): cap2 = {cap2}")

        to_delete = [(x, u) for _ in range(cap2)] + [(x, v2)
                                                     for _ in range(cap2)]
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
        if connectivity(h, indicator) < req:
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


def lovasz_decomposition(G, req, verbose=False):
    r"""
    Consider that the given requirement is of the form
    `2k` for some `k \geq 1`. Then this function
    computes the decomposition of a the `2k`-connected
    graph `G` into a pair of vertices and `2k` edges
    connecting them.

    INPUT:

    - `G` -- a Graph.
    - `req` -- the global edge-connectivity requirement.
    - `verbose` -- if the user wants debugging output.

    OUTPUT: `(H, (A, R))` where `H` is a Graph with only
    a pair of vertices and `req` edges connecting them.

    EXAMPLES:

    Example usage::

        sage: from orientations import lovasz_decomposition

    TEST:

    We know that, for even n, all vertices at `g = K(n)` have
    edge-connectivity `n - 1`. Hencem the decomposed graph
    should have `n - 1` edges between the two vertice::

        sage: g = graphs.CompleteGraph(7)
        sage: g.allow_multiple_edges(True)
        sage: req, k = 6, 3

    By the Lovasz decomposition Theorem, h should have 2
    vertices and 2k edges connecting those vertices::

        sage: h, _ = lovasz_decomposition(g, 6)
        sage: assert len(h) == 2 and len(h.edges()) == 2 * k
    """

    if not G.allows_multiple_edges():
        raise TypeError('should allow multiple edges')

    add, rm, rm_v = [], [], []

    step = 0
    while len(G) > 2:
        # Remove unneded edges (minimally edge connected)
        edges = G.edges()

        for e in edges:
            G.delete_edge(e)
            if connectivity(G) < req:
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
            indicator = _get_indicator(G, [x, u])
            cap = splitting_off_to_capacity(
                G, x, indicator, req, candidates=(u, u), verbose=verbose)

            if cap > 0:
                if verbose:
                    print(
                        f"pre split-off ({x}-{u}, {x}-{u}) to cap = {cap} ind = {indicator}")
                    G.plot(layout='circular').show()

                # Remove the pertinent edges
                to_delete = [(x, u) for _ in range(cap)] + [(x, u)
                                                            for _ in range(cap)]
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
            step += 1
            continue

        # Try to split-off to capacity x
        # using the Lemma 3.2 from efficient edge split.
        C, added, removed = complete_splitting_off(G, x, req, verbose=verbose)

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
        if connectivity(G) < req:
            G.add_edge(e)
            break

        add.insert(0, [])
        rm.insert(0, [e])
        rm_v.insert(0, None)

    return G, (add, rm, rm_v)


def orientation(G, k, verbose=False):
    r"""
    Computes an arbitrary `k`-connected orientation
    of the graph `G`.

    INPUT:

    - `G` -- a Graph.
    - `k` -- the wanted connectivity.
    - `verbose` -- if the user wants debugging output.

    OUTPUT: A `k`-connected orientation of `G`.

    EXAMPLES:

    Example usage::

        sage: from orientations import orientation

    TEST:

    We know that, for even n, all vertices at `g = K(n)` have
    edge-connectivity `n - 1`::

        sage: g = graphs.CompleteGraph(7)
        sage: g.allow_multiple_edges(True)
        sage: req, k = 6, 3

    Now let's actually generate a `k`-connected
    orientation of `g`::

        sage: ori = orientation(g.copy(), k)

    The following properties should be true:
    1. The undirected version of ori should be g
    2. ori should be k-connected::

        sage: assert ori.to_undirected() == g

    An orientation of `K_n` should not have multiple edges::

        sage: assert ori.has_multiple_edges() == False
        sage: ori.allow_multiple_edges(False)
        sage: assert ori.edge_connectivity() == k
    """

    # Make sure we allow multiple edges
    G.allow_multiple_edges(True)

    req = 2 * k
    reduced_G, ops = lovasz_decomposition(G.copy(), req, verbose=verbose)

    # Generate random k-connected orientation
    reduced_G.to_directed()

    from sage.graphs.digraph import DiGraph
    g = DiGraph(data=[reduced_G.vertices(), []],
                format='vertices_and_edges',
                multiedges=reduced_G.allows_multiple_edges(),
                loops=reduced_G.allows_loops(),
                weighted=reduced_G.weighted(),
                pos=reduced_G.get_pos(),
                name=f'Random {k}-connected orientation of {reduced_G.name()}')

    if hasattr(reduced_G, '_embedding'):
        from copy import copy
        g._embedding = copy(reduced_G._embedding)

    edges = reduced_G.edges()

    from random import shuffle
    looper = [i for i in range(len(edges))]
    shuffle(looper)

    for i in looper:
        u, v, label = edges[i][0], edges[i][1], edges[i][2]

        if i % 2:
            g.add_edge(u, v, label)
        else:
            g.add_edge(v, u, label)

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


def _outdegree_sequence(D):
    r"""
    Returns the outdegree sequence of a digraph.

    INPUT:

    - `D` -- a DiGraph.

    OUTPUTS: The outdegree sequence of `D`.

    EXAMPLES:

    Example usage::

        sage: from orientations.orientations import _outdegree_sequence
        sage: _outdegree_sequence(graphs.CompleteGraph(5))
        [4, 4, 4, 4, 4]

    TEST::

        sage: _outdegree_sequence(graphs.CompleteGraph(5))
        [4, 4, 4, 4, 4]
    """

    out_degree = []

    for i, v in enumerate(D):
        out_degree.append(0)

        for u in D.neighbor_iterator(v):
            if D.has_edge(v, u):
                out_degree[i] += 1

    return out_degree


def _bfs(G, u, v):
    r"""
    Computes the BFS path from `u` to `v` and
    returns the reverse.

    INPUT:

    - `G` -- a Graph.
    - `u` -- a vertex.
    - `v` -- a vertex.

    OUTPUTS: The `u-v` path in `G` but reversed.

    EXAMPLES:

    Example usage::

        sage: from orientations.orientations import _bfs

    TEST::

        sage: g = DiGraph({1: [2], 2: [3, 4]})
        sage: path = _bfs(g, 1, 4)
        sage: assert path == [4, 2, 1]
    """

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


def _reverse(D, u, v):
    r"""
    Computes the BFS path from `u` to `v` and
    reverses it in `D`.

    INPUT:

    - `D` -- a DiGraph.
    - `u` -- a vertex.
    - `v` -- a vertex.

    OUTPUTS: The `u-v` path in `G` but reversed.

    EXAMPLES:

    Example usage::

        sage: from orientations.orientations import _reverse

    TEST:

    Check that the path was actually reversed in
    the original graph::

        sage: D = DiGraph({1: [2], 2: [3, 4]})
        sage: path = _reverse(D, 1, 4)
        sage: assert path == [4, 2, 1]
        sage: for u, v in ((1, 2), (2, 4)):
        ....:     assert not D.has_edge(u, v) and D.has_edge(v, u)
    """
    path = _bfs(D, u, v)

    # Reverse path and re-iterate
    for i in range(len(path)-1):
        x, y = path[i], path[i+1]
        D.delete_edge(y, x)
        D.add_edge((x, y))

    return path


def _is_flippable(D, u, v, req, step=0):
    """
    O(km) algorithm to check if the path from `u` to `v` is
    flippable. I.e. we can reverse it up to req + 1 times.

    INPUT:

    - `D` -- a DiGraph.
    - `u` -- a vertex.
    - `v` -- a vertex.
    - `req` -- the connectivity requirement.

    OUTPUTS: The `u-v` path in `G` but reversed.

    EXAMPLES:

    Example usage::

        sage: from orientations.orientations import _is_flippable

    TEST:

    Compute a 3-connected orientation of K_7::

        sage: from orientations import orientation
        sage: d = orientation(graphs.CompleteGraph(7), 3)
        sage: d.allow_multiple_edges(False)
        sage: assert d.edge_connectivity() == 3

    As d is 3-connected, we should be able to flip every vertex pair for
    k = 1 (resp. 2), as it means we can find 2 (resp. 3) edge-disjoint
    paths for every pair. Also, we should not be able to flip any pair
    with k = 3::

        sage: assert all(_is_flippable(d.copy(), u, v, 1) \
                for u, v in Subsets(d.vertices(), 2))
        sage: assert all(_is_flippable(d.copy(), u, v, 2) \
                for u, v in Subsets(d.vertices(), 2))
        sage: assert all(not _is_flippable(d.copy(), u, v, 3) \
                for u, v in Subsets(d.vertices(), 2))
    """

    # If we applied this procedure req + 1 times
    # then the original (u, v) is flippable
    # i.e. at most we will recurse req = k times
    if step > req:
        return True

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
    return _is_flippable(D, u, v, req=req, step=step+1)


def _reverse_path_neg(D, F, v, req, seen=[], verbose=False):
    r"""
    Helper function for the outdegree interator.
    Reverses the path, in negative orientation, `F` in `D`...

    INPUT:

    - `D` -- a DiGraph.
    - `F` -- a set of vertices.
    - `v` -- a vertex.
    - `req` -- the edge-connectivity requirement.
    - `seen` -- ...
    - `verbose` -- if the user wants debugging output.

    OUTPUT: An iterator.

    TEST::

        sage: from orientations.orientations import _reverse_path_neg
    """
    if verbose:
        print(
            f"Reverse-: F = {F} and v = {v} and req = {req} and seen = {seen}")

    candidate = None
    for u in D:
        if u == v:
            continue

        if u not in F and u and _is_flippable(D.copy(), v, u, req=req):
            candidate = u
            break

    if candidate is not None:
        path = _reverse(D, v, candidate)
        if verbose:
            print(f"Reverse- ({candidate}) path: {path}")

        yield from _reverse_path_neg(D.copy(), F, v, req, seen=seen+[candidate], verbose=verbose)
        yield from outdegree_sequence_iterator(D.copy(), F + [v], req, verbose=verbose)
    else:
        if verbose:
            print(f"Reverse-: candidate is None")
        pass


def _reverse_path_pos(D, F, v, req, seen=[], verbose=False):
    r"""
    Helper function for the outdegree interator.
    Reverses the path, in positive orientation, `F` in `D`...

    INPUT:

    - `D` -- a DiGraph.
    - `F` -- a set of vertices.
    - `v` -- a vertex.
    - `req` -- the edge-connectivity requirement.
    - `seen` -- ...
    - `verbose` -- if the user wants debugging output.

    OUTPUT: An iterator.

    TEST::

        sage: from orientations.orientations import _reverse_path_pos
    """

    if verbose:
        print(
            f"Reverse+: F = {F} and v = {v} and req = {req} and seen = {seen}")

    candidate = None
    for u in D:
        if u == v:
            continue

        if u not in F and u and _is_flippable(D.copy(), u, v, req=req):
            candidate = u
            break

    if candidate is not None:
        path = _reverse(D, candidate, v)
        if verbose:
            print(f"Reverse+ ({candidate}) path: {path}")

        yield from _reverse_path_pos(D.copy(), F, v, req, seen=seen+[candidate], verbose=verbose)
        yield from outdegree_sequence_iterator(D.copy(), F + [v], req, verbose=verbose)
    else:
        if verbose:
            print(f"Reverse+: candidate is None")
        pass


def alpha_orientations_iterator(D, F=[], verbose=False):
    r"""
    Return an iterator over all `\alpha`-orientations of `D`.

    INPUT:

    - `D` -- a DiGraph.
    - `F` -- a set of vertices.
    - `verbose` -- if the user wants debugging output.

    OUTPUT: An iterator.

    EXAMPLES:

    Example usage::

        sage: from orientations import alpha_orientations_iterator

    TEST:
    """

    from collections import defaultdict

    # Compute the multiplicity of D
    mults_F = defaultdict(lambda: defaultdict(lambda: 0))
    for u, v, _ in F:
        mults_F[u][v] += 1

    a = None

    # Pick a random edge from A
    for e in D.edge_iterator():
        u, v, _ = e

        if mults_F[u][v] == 0:
            a = e
            break

    if a is not None:
        if verbose:
            print(f"EnOPODS: F = {F}, a = {a}")

        u, v, _ = a

        # Call EnOPODS with the edge a in F
        yield from alpha_orientations_iterator(D.copy(), F + [a], verbose=verbose)

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

            yield from alpha_orientations_iterator(D.copy(), F + [(a[1], a[0], a[2])], verbose=verbose)

    else:
        if verbose:
            print(f"EnOPODS: yielding!")
        yield D.copy()


def _differ(A, B):
    r"""
    Check if the iterable A differ from B.
    As fast as possible.

    INPUT:

    - `A` -- an iterable.
    - `B` -- an iterable.

    OUTPUT: If A differs from B.

    EXAMPLES:

    Example usage::

        sage: from orientations.orientations import _differ
        sage: not _differ([], [])
        True
        sage: _differ([1, 2], [1, 2, 3])
        True

    TEST::

        sage: assert not _differ([], [])
        sage: assert _differ([1, 2], [1, 2, 3])
        sage: assert _differ([1, 2, 4], [1, 2, 3])
    """
    if len(A) != len(B):
        return True

    if len(A) == 0:
        return False

    different = False

    for a, b in zip(A, B):
        if a not in B or b not in A:
            different = True
            break

    if different:
        return True

    return False


def outdegree_sequence_iterator(D, F, req, verbose=False):
    r"""
    Return an iterator over all `k`-connected orientations of `D`.

    INPUT:

    - `D` -- a DiGraph.
    - `F` -- a set.
    - `req` -- the edge-connectivity requirement, `k`.
    - `verbose` -- if the user wants debugging output.

    OUTPUT: An iterator.

    EXAMPLES:

    Example usage::

        sage: from orientations import outdegree_sequence_iterator

    TEST:
    """

    v = None

    for w in D:
        if w not in F:
            v = w
            break

    if v is not None:
        if verbose:
            print(f"EnODS2: enter with F = {F} and v = {v}")

        yield from _reverse_path_pos(D.copy(), F, v, req, verbose=verbose)
        yield from _reverse_path_neg(D.copy(), F, v, req, verbose=verbose)
        yield from outdegree_sequence_iterator(D.copy(), F + [v], req, verbose=verbose)
    else:
        # Case then we want all
        if verbose:
            print(
                f"EnODS2: yielding from EnOPODS with {_outdegree_sequence(D)}")
        yield from alpha_orientations_iterator(D.copy(), [], verbose=verbose)


def k_orientations_iterator(G, k, verbose=False):
    r"""
    Return an iterator over all `k`-connected orientations of
    an undirected multi-graph `G`.

    INPUT:

    - `G` -- a Graph, which may have multiple edges.
    - `k` -- the wanted connectivity of the orientations.
    - `verbose` -- if the user wants debugging output.

    OUTPUT: An iterator over all `k`-connected orientations of `G`.

    EXAMPLES:

    If `k = 1` we get the same results as the strong orientations iterator,
    but we need to multiply by two as it does not consider reflections::

        sage: from orientations import k_orientations_iterator
        sage: from sage.graphs.orientations import strong_orientations_iterator
        sage: pet = graphs.PetersenGraph()
        sage: strong = len(list(strong_orientations_iterator(pet))) * 2
        sage: strong
        1920
        sage: new = len(list(k_orientations_iterator(pet, 1)))
        sage: assert strong == new

    We can compute how many 2-connected have the Harary `H_{4,7}` graph::

        sage: h = graphs.HararyGraph(4, 7)
        sage: len(list(k_orientations_iterator(h, 2)))
        60

    TEST:

    For k = 1 we should get the same result as the
    native strong orientations iterator from Sage::

        sage: for n in (3, 5):
        ....:     g = graphs.CompleteGraph(n)
        ....:     new = len(list(k_orientations_iterator(g, 1)))
        ....:     strong = len(list(strong_orientations_iterator(g))) * 2
        ....:     assert new == strong

    Now let's test for some random graphs with known connectivities::

        sage: query = GraphQuery(
        ....:     display_cols=['num_vertices', 'edge_connectivity'],
        ....:     edge_connectivity=['=', 4],
        ....:     num_vertices=['<', 7],
        ....:     num_edges=['<', 15],
        ....: )
        sage: items = query.get_graphs_list()
        sage: for item in items[:3]:
        ....:     g = item.copy()
        ....:     g.allow_multiple_edges(True)
        ....:     got_1, got_2 = 0, 0
        ....:     for ori in strong_orientations_iterator(g):
        ....:         got_1 += 1
        ....:         if ori.edge_connectivity() == 2:
        ....:             got_2 += 1
        ....:     got_1 *= 2
        ....:     got_2 *= 2
        ....:     want_1 = len(list(k_orientations_iterator(g.copy(), 1)))
        ....:     want_2 = len(list(k_orientations_iterator(g.copy(), 2)))
        ....:     assert got_1 == want_1 and got_2 == want_2

    We know that this graph has 3842 2-connected orientations
    (see Conclusions arXiv:1908.02050)::

        sage: g1 = {
        ....:     1: [2, 3, 8, 9],
        ....:     2: [3, 4, 8],
        ....:     3: [4, 9],
        ....:     4: [5, 6, 8, 9],
        ....:     5: [6, 7, 8],
        ....:     6: [7, 9],
        ....:     7: [8, 9],
        ....:     8: [9],
        ....: }
        sage: oris = len(list(k_orientations_iterator(Graph(g1), 2)))
        sage: assert oris == 3842
    """

    D = orientation(G.copy(), k, verbose=False)

    if verbose:
        print(f'od_seq: {_outdegree_sequence(D)}')

    return outdegree_sequence_iterator(D.copy(), [], k, verbose=verbose)
