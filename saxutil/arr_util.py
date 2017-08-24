def detect_dup_indices(sorted_arr):
    dup_indices = set()
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] == sorted_arr[i - 1]:
            dup_indices.add(i - 1)
            dup_indices.add(i)
    return sorted(list(dup_indices))


def detect_dup_vals(arr):
    arr.sort()
    indices = detect_dup_indices(arr)
    return set([arr[i] for i in indices])
