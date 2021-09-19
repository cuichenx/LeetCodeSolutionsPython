import bisect
import math
import time
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List

from helpers import print_assert
import heapq
import numpy as np

class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                if abs(nums[i] - nums[j]) == k:
                    count += 1
        return count

    def findOriginalArray(self, changed: List[int]) -> List[int]:
        cntr = Counter(changed)
        ret = []
        done = Counter()
        for num in sorted(changed):
            if done[num] > 0:
                done[num] -= 1
                continue
            if cntr[num*2] > 0:
                cntr[num] -= 1
                cntr[num*2] -= 1
                ret.append(num)
                done[num*2] += 1
            else:
                return []
        if sum(cntr.values()) == 0:
            return ret
        else:
            return []

    def test2(self):
        print_assert(self.findOriginalArray([1, 3, 4, 2, 6, 8]), [1, 3, 4])
        print_assert(self.findOriginalArray([1, 1, 3, 4, 8, 2, 6, 8, 4, 2]), [1, 1, 3, 4, 4])
        print_assert(self.findOriginalArray([6, 3, 0, 1]), [])
        print_assert(self.findOriginalArray([1]), [])

    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        rides_by_start = defaultdict(list)
        for (start, end, tip) in rides:
            rides_by_start[start].append([end, tip])

        dp = [0] * (n+1)  # max profit you can get when you are at location [0, n-1]
        for k in range(n-1, 0, -1):
            max_profit = dp[k+1]  # simply drive to the next location
            for (end, tip) in rides_by_start[k]:
                max_profit = max(max_profit, end - k + tip + dp[end])
            dp[k] = max_profit

        return dp[1]


    def test3(self):
        print_assert(self.maxTaxiEarnings(5, [[2,5,4],[1,5,1]]), 7)
        print_assert(self.maxTaxiEarnings(20, [[1,6,1],[3,10,2],[10,12,3],[11,12,2],[12,15,2],[13,18,1]]), 20)

    def minOperations(self, nums: List[int]) -> int:
        diffs = []
        nums.sort()
        n = len(nums)
        for i in range(n-1):
            if nums[i+1] - nums[i] > 0:
                diffs.append(nums[i+1] - nums[i])
        return n - 1 - self.longest_sublist_le_k(diffs, n-1)

    def longest_sublist_le_k(self, nums, k):
        i = 0
        max_len = 0
        window_sum = 0
        # following hint, for each j, find the largest window (left most i) whose product is less than k
        for j in range(len(nums)):
            window_sum += nums[j]  # add nums[j] to window
            while window_sum > k and i <= j:
                window_sum -= nums[i]  # remove nums[i] from window
                i += 1
            # now window_prod < k, or i = j+1
            max_len = max(max_len, (j + 1 - i))
        return max_len

    def test4(self):
        print_assert(self.longest_sublist_le_k([11, 4, 6, 2, 5, 3, 10, 1], 9), 2)
        print_assert(self.longest_sublist_le_k([9, 99, 999], 3), 0)
        print_assert(self.minOperations([4, 2, 5, 3]), 0)
        print_assert(self.minOperations([1,2,3,5,6]), 1)
        print_assert(self.minOperations([1,10,100,1000]), 3)



if __name__ == '__main__':
    Solution().test4()


