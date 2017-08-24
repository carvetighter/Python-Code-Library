def do_ranges_overlap(b0, e0, b1, e1):
    """
    b0 and e0 are one pair of start and end boundaries
    b1 and e1 are another pair of start and end boundaries.
    all four parameters are integers

    If the range from b0 to e0 overlaps with the range from b1 to e1
    this method returns True. Otherwise it returns False.

    The following statements are True
    - do_ranges_overlap(1, 5, 2, 4)
    - do_ranges_overlap(1, 3, 2, 3)

    The following statements are False
    - do_ranges_overlap(1, 2, 3, 4)
    - do_ranges_overlap(0, 4, 6, 9)
    """
    if b0 <= b1 <= e0 or b0 <= e1 <= e0:
        return True
    return False


def is_subsumed(b0, e0, b1, e1):
    """
    Is range b0, e0 subsumed by range b1, e1?
    If yes, return True
    If no, return False
    """
    if not do_ranges_overlap(b0, e0, b1, e1):
        return False
    size0 = e0 - b0
    size1 = e1 - b1
    if size0 < size1:
        return True
    return False


def subsumed_bound_indices(bnds):
    """
    Given a list of integer pairs representing boundaries, find all pair
    indices where the numeric range is completely subsumed by the numeric
    range of another pair in the list.

    :type bnds: list<tuple(int, int,)>
    :rtype: set<int>
    """
    indices = set()
    for i in range(len(bnds)):
        for j in range(len(bnds)):
            assert(bnds[i][0] <= bnds[i][1] and bnds[j][0] <= bnds[j][1])
            if i == j:
                continue
            if is_subsumed(bnds[i][0], bnds[i][1], bnds[j][0], bnds[j][1]):
                indices.add(i)
            elif is_subsumed(bnds[j][0], bnds[j][1], bnds[i][0], bnds[i][1]):
                indices.add(j)
    return indices