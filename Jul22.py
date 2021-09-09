import bisect
from bisect import bisect_left
from itertools import accumulate
from typing import List

from helpers import print_assert


class Solution:
    # 915. Partition Array into Disjoint Intervals
    # Given an array nums, partition it into two (contiguous) subarrays left and right so that:
    #
    # Every element in left is less than or equal to every element in right.
    # left and right are non-empty.
    # left has the smallest possible size.
    # Return the length of left after such a partitioning.  It is guaranteed that such a partitioning exists.
    def partitionDisjoint(self, nums: List[int]) -> int:

        cur_max_idx = 0
        left_max_idx = 0
        res = 1

        for i in range(len(nums)):
            if nums[i] < nums[left_max_idx]:
                # found a dip, set divider to after this element
                left_max_idx = cur_max_idx
                res = i+1
            elif nums[i] > nums[cur_max_idx]:
                # update current max
                cur_max_idx = i

        return res

    # 363. Max Sum of Rectangle No Larger Than K
    # Given an m x n matrix matrix and an integer k, return the max sum of a rectangle in the matrix such that its sum is no larger than k.
    #
    # It is guaranteed that there will be a rectangle with a sum no larger than k.
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        # 2D Kadane algorithm
        # assume matrix is tall and thin. We apply Kadane algorithm on each subarray in each row
        m, n = len(matrix), len(matrix[0])
        global_max = -float('inf')
        for left in range(n):
            sums = [0] * m
            for right in range(left, n):
                # sum matrix[row][left:right] for each row, then apply kadane
                for row in range(m):
                    sums[row] += matrix[row][right]
                local_max = self.kadaneMaxK(sums, k)
                if local_max == k:
                    return k
                elif global_max < local_max < k:
                    global_max = local_max

        return global_max

    def kadaneMaxK(self, arr: List[int], k) -> int:
        # global_max = -float('inf')
        # local_max = 0
        # for i in range(len(arr)):
        #     local_max = max(local_max + arr[i], arr[i])
        #     if local_max == k:
        #         return k
        #     elif global_max < local_max < k:
        #         global_max = local_max
        #
        # return global_max
        SList, ans = [0], -float("inf")
        for s in accumulate(arr):
            idx = bisect_left(SList, s - k)
            if idx < len(SList):
                ans = max(ans, s - SList[idx])
            bisect.insort(SList, s)
        return ans

if __name__ == '__main__':
    sol = Solution()
    # print_assert(actual=sol.partitionDisjoint([5,0,3,8,6]), expected=3)
    # print_assert(actual=sol.partitionDisjoint([1,1,1,0,6,12]), expected=4)
    # print_assert(actual=sol.partitionDisjoint([1, 2]), expected=1)
    # print_assert(actual=sol.partitionDisjoint([1, 2, 3]), expected=1)
    # print_assert(actual=sol.partitionDisjoint([1, 3, 2]), expected=1)
    # print_assert(actual=sol.partitionDisjoint([2, 1, 3]), expected=2)
    # print_assert(actual=sol.partitionDisjoint([26, 51, 40, 58, 42, 76, 30, 48, 79, 91]), expected=1)
    # print_assert(actual=sol.partitionDisjoint([51, 40, 58, 42, 76, 30, 48, 79, 91]), expected=7)
    # print_assert(actual=sol.partitionDisjoint([40, 58, 42, 76, 30, 48, 79, 91]), expected=6)
    # print_assert(actual=sol.partitionDisjoint([1, 1]), expected=1)
    # print_assert(actual=sol.partitionDisjoint([3,1,8,4,9,7,12,0,0,12,6,12,6,19,24,90,87,54,92,60,31,59,75,90,20,38,52,
    #                                            51,74,70,86,20,27,91,55,47,54,86,15,16,74,32,68,27,19,54,13,22,34,74,76,
    #                                            50,74,97,87,42,58,95,17,93,39,33,22,87,96,90,71,22,48,46,37,18,17,65,54,
    #                                            82,54,29,27,68,53,89,23,12,90,98,42,87,91,23,72,35,14,58,62,79,30,67,44,48]),
    #              expected=13)
    print_assert(actual=sol.maxSumSubmatrix([[1, 0, 1], [0, -2, 3]], 2), expected=2)
    print_assert(actual=sol.maxSumSubmatrix([[2, 2, -1]], 3), expected=3)
    print_assert(actual=sol.maxSumSubmatrix(
        [[6, -5, -7, 4, -4],
         [-9, 3, -6, 5, 2],
         [-10, 4, 7, -6, 3],
         [-8, 9, -3, 3, -7]]
        , 99), expected=17)
    print_assert(actual=sol.maxSumSubmatrix(
        [[6, -5, -7, 4, -4],
         [-9, 3, -6, 5, 2],
         [-10, 4, 7, -6, 3],
         [-8, 9, -3, 3, -7]]
        , 17), expected=17)
    print_assert(actual=sol.maxSumSubmatrix(
        [[6, -5, -7, 4, -4],
         [-9, 3, -6, 5, 2],
         [-10, 4, 7, -6, 3],
         [-8, 9, -3, 3, -7]]
        , 16), expected=16)
    print_assert(actual=sol.maxSumSubmatrix(
        [[6, -5, -7, 4, -4],
         [-9, 3, -6, 5, 2],
         [-10, 4, 7, -6, 3],
         [-8, 9, -3, 3, -7]]
        , 15), expected=14)
    print_assert(actual=sol.maxSumSubmatrix([[2, 2, -1]], 0), expected=-1)
    print_assert(actual=sol.maxSumSubmatrix([[5, -4, -3, 4],
                                             [-3, -4, 4, 5],
                                             [5, 1, 5, -4]], 10), expected=10)
    print_assert(actual=sol.maxSumSubmatrix([[5, -4, -3, 4],
                                             [-3, -4, 4, 5],
                                             [5, 1, 5, -4]], 8), expected=8)