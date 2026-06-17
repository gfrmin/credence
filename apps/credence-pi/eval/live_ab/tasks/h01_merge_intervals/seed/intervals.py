def merge(intervals):
    """Merge all overlapping or touching intervals.

    Each interval is a [start, end] list with start <= end. Return the merged
    intervals sorted by start, with no two overlapping or touching.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        # BUG 1: strict < drops touching intervals ([1,2],[2,3] should merge).
        # BUG 2: assigning end clobbers a larger existing end (nested [1,5],[2,3]).
        if start < merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged
