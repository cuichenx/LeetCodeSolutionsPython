from collections import defaultdict
from typing import List

from helpers import print_assert


class Solution:
    # 16. 3Sum Closest
    #     Given an array nums of n integers and an integer target,
    #     find three integers in nums such that the sum is closest to target.
    #     Return the sum of the three integers. You may assume that each input would have exactly one solution
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        min_diff = abs(nums[0]+nums[1]+nums[2] - target)
        res = nums[0]+nums[1]+nums[2]
        for i in range(n-2):
            k = n-1
            for j in range(i+1, k):
                while k > j:
                    s = nums[i] + nums[j] + nums[k]
                    diff = s - target
                    if diff >= min_diff:
                        # s too large, decrease k
                        k -= 1
                    elif -min_diff < diff < min_diff:
                        # in range
                        res = s
                        min_diff = abs(diff)
                        if diff > 0:  # if diff is positive, try to go even smaller
                            k -= 1
                        else: break   # else try to go a bit larger
                    else:  # diff <= -min_diff
                        # too small, increase j
                        break

        return res
    # Time: O(n^2), Space: space required for sorting, O(n) in python

    # 560. Subarray Sum Equals K
    # Given an array of integers nums and an integer k, return the total number of continuous subarrays
    # whose sum equals to k.
    def subarraySum(self, nums: List[int], k: int) -> int:
        cumul_sum = 0
        sum_count = {0: 1}
        c = 0
        for i in range(len(nums)):
            cumul_sum += nums[i]
            c += sum_count.get(cumul_sum - k, 0)
            sum_count[cumul_sum] = sum_count.get(cumul_sum, 0) + 1

        return c
    # Time: O(n), Space: O(n)

if __name__ == '__main__':
    sol = Solution()
    # print_assert(actual=sol.threeSumClosest([-1, 2, 1, -4], 1), expected=2)
    # print_assert(actual=sol.threeSumClosest([1, 4, 2], 1), expected=7)
    # print_assert(actual=sol.threeSumClosest([1, 4, 2], 99), expected=7)
    # print_assert(actual=sol.threeSumClosest([1, 4, 2, 8], 10), expected=11)
    # print_assert(actual=sol.threeSumClosest([1, 4, 2, 8], 14), expected=14)
    # print_assert(actual=sol.threeSumClosest([1, 4, 2, 8, 3, 9], 6), expected=6)
    # print_assert(actual=sol.threeSumClosest([1, 4, 2, 8, 3, 9], 8), expected=8)
    # print_assert(actual=sol.threeSumClosest([1, 1, 1, 1], 0), expected=3)
    print_assert(actual=sol.subarraySum([1, 1, 1], 2), expected=2)
    print_assert(actual=sol.subarraySum([1, 2, 3], 3), expected=2)
    print_assert(actual=sol.subarraySum([3, 2, 1, 8, 7, 6, 5, 4], 9), expected=2)
    print_assert(actual=sol.subarraySum([1], 0), expected=0)
    print_assert(actual=sol.subarraySum([-1, -1, 1], 0), expected=1)

