import bisect
import math
import time
from collections import defaultdict
from typing import List

from helpers import print_assert
import heapq


class AlgoIIDay1:
    # 34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        idx1 = bisect.bisect_left(nums, target)
        if idx1 < len(nums) and nums[idx1] == target:
            idx2 = bisect.bisect_right(nums, target)
            return [idx1, idx2 - 1]
        else:
            return [-1, -1]

    def test1(self):
        print_assert(self.searchRange([5, 7, 7, 8, 8, 10], 8), [3, 4])
        print_assert(self.searchRange([5, 7, 7, 8, 8, 10], 6), [-1, -1])
        print_assert(self.searchRange([], 0), [-1, -1])

    # 33. Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        # two binary searches
        # first one to locate the minimum element (orig start)
        # second one to do normal binary search, given adjusted indices
        n = len(nums)
        lo, hi = -1, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if nums[mid] <= nums[hi]:
                # second half is sorted.
                hi = mid
            else:
                # first half is sorted
                lo = mid
        orig_start = hi

        # print(orig_start)
        def f(x):
            # orig index in sorted array to new index in rotated array
            return (x + orig_start) % n

        lo, hi = 0, n  # lo mid hi are rotated indices
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= nums[f(mid)]:
                hi = mid
            else:
                lo = mid + 1
        if nums[f(lo)] == target:
            return f(lo)
        else:
            return -1

    def test2(self):
        print_assert(self.search([0], 0), 0)
        print_assert(self.search([0, 1], 0), 0)
        print_assert(self.search([0, 1, 2], 0), 0)
        print_assert(self.search([0, 1, 2, 3], 0), 0)
        print_assert(self.search([0, 1, 2, 3, 4], 0), 0)
        print_assert(self.search([4, 5, 6, 7, 0, 1, 2], 3), -1)
        print_assert(self.search([3, 1], 1), 1)
        print_assert(self.search([1, 3], 1), 0)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], 3), 10)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4], 3), 9)
        print_assert(self.search([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], 3), 8)
        print_assert(self.search([5, 6, 7, 8, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 0, 1, 2, 3, 4], 3), 7)
        print_assert(self.search([5, 6, 7, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 0, 1, 2, 3, 4], 3), 6)
        print_assert(self.search([5, 6, 0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([5, 6, 0, 1, 2, 3, 4], 3), 5)
        print_assert(self.search([5, 0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([5, 0, 1, 2, 3, 4], 3), 4)
        print_assert(self.search([0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([0, 1, 2, 3, 4], 3), 3)

    # 74. Search a 2D Matrix
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])

        lo, hi = 0, m*n  # lo mid hi are rotated indices
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= matrix[mid//n][mid%n]:
                hi = mid
            else:
                lo = mid + 1
        return lo < m*n and matrix[lo//n][lo%n] == target

    def test3(self):
        print_assert(self.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3), True)
        print_assert(self.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13), False)
        print_assert(self.searchMatrix([[1]], 1), True)
        print_assert(self.searchMatrix([[1]], 2), False)

if __name__ == '__main__':
    AlgoIIDay1().test1()
    AlgoIIDay1().test2()
    AlgoIIDay1().test3()
