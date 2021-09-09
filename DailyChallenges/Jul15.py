from typing import List
from bisect import bisect_left, bisect_right

class Solution:

    # 611
    # Given an integer array nums, return the number of triplets chosen from the array that can make triangles
    # if we take them as side lengths of a triangle.
    def triangleNumber(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return 0
        c = 0
        sorted_nums = [x for x in sorted(nums) if x>0]
        n = len(sorted_nums)
        for i in range(n-2):
            k = i + 2
            for j in range(i+1, n-1):
                # for k in range(j+1, n)
                # find the range of numbers valid for triangle
                # range_left = bisect_right(sorted_nums, sorted_nums[j]-sorted_nums[i], lo=j+1)  always 0
                # range_right = bisect_left(sorted_nums, sorted_nums[j]+sorted_nums[i], lo=j+1)
                # c += range_right - (j+1)
                while k < n:
                    if sorted_nums[i] + sorted_nums[j] > sorted_nums[k]:
                        k += 1
                    else:
                        break
                # k-1 is the last index admissible. j+1 is the first index admissible
                c += k - j - 1


        return c
    # time: O(n^2 log(n) )
    # space: O(n)


    # 1546
    # Given an array nums and an integer target.
    # Return the maximum number of non-empty non-overlapping subarrays such that the sum of values in each subarray
    # is equal to target.
    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        # greedy algorithm. start from beginning, find the first occurrence (ordered by end index)
        # of subarray summing to target. start from the end of that subarray to continue searching
        c = 0
        # define a partial sum (psum) as the sum from the first element to the current element
        # hence, the sum of a subarray can be written as the difference between two psums
        psum = 0
        previous_psums = {0}

        for num in nums:
            psum += num
            if psum - target in previous_psums:  # if psum - one of previous psums == target
                # bingo
                c += 1
                previous_psums = {psum}  # all the recoreded psums would overlap with the current found one, so reset
            else:
                previous_psums.add(psum)

        return c

    def maxNonOverlapping_discussion(self, nums: List[int], target: int) -> int:
        f = {0}
        s = 0
        ans = 0
        for j in nums:
            s += j
            if s - target in f:
                ans += 1
                f = set([])
            f.add(s)
        return ans

if __name__ == '__main__':
    sol = Solution()
    # print(sol.triangleNumber([2, 2, 3, 4]), 3)
    # print(sol.triangleNumber([4, 2, 3, 4]), 4)
    # print(sol.triangleNumber([1]), 0)
    # print(sol.triangleNumber([1, 2]), 0)
    # print(sol.triangleNumber([1, 2, 3]), 0)
    # print(sol.triangleNumber([1, 2, 4]), 0)
    # print(sol.triangleNumber([1, 2, 2]), 1)
    # print(sol.triangleNumber([1, 2, 3, 4, 5, 6, 7]), 13)
    # print(sol.triangleNumber([0, 0, 0]), 0)
    # print(sol.triangleNumber([1, 1, 1]), 1)
    print(sol.maxNonOverlapping([-1, 3, 5, 1, 4, 2, -9], 6), 2)
    print(sol.maxNonOverlapping([1, 1, 1, 1, 1], 2), 2)
    print(sol.maxNonOverlapping([-2, 6, 6, 3, 5, 4, 1, 2, 8], 10), 3)
    print(sol.maxNonOverlapping([0, 0, 0], 0), 3)