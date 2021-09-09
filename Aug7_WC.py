import bisect
import math
from typing import List

from helpers import print_assert
import heapq
import numpy as np

class Solution:
    def isPrefixString(self, s: str, words: List[str]) -> bool:
        for w in words:
            if s.startswith(w):
                s = s[len(w):]
                if s == '':
                    return True
            else:
                return False

    def test1(self):
        print_assert(self.isPrefixString("iloveleetcode", ["i","love","leetcode","apples"]), True)
        print_assert(self.isPrefixString("iloveleetcode", ['apples', "i","love","leetcode","apples"]), False)

    def minStoneSum(self, piles: List[int], k: int) -> int:
        neg_piles = [-p for p in piles]
        heapq.heapify(neg_piles)
        for _ in range(k):
            biggest = -heapq.heappop(neg_piles)
            heapq.heappush(neg_piles, -math.ceil(biggest/2))
        return -sum(neg_piles)

    def test2(self):
        print_assert(self.minStoneSum([5, 4, 9], 2), 12)
        print_assert(self.minStoneSum([4, 3, 6, 7], 3), 12)

    def minSwaps(self, s: str) -> int:
        # 1. find max negative balance
        bal = 0
        most_neg_bal = 0
        for c in s:
            bal += 1 if c == '[' else -1
            most_neg_bal = min(bal, most_neg_bal)

        return math.ceil(-most_neg_bal/2)

    def test3(self):
        print_assert(self.minSwaps("][]["), 1)
        print_assert(self.minSwaps("]]][[["), 2)
        print_assert(self.minSwaps("[]"), 0)

    def longestObstacleCourseAtEachPosition_slow(self, obstacles: List[int]) -> List[int]:
        ret = [1]
        for i in range(1, len(obstacles)):
            if obstacles[i] < obstacles[i-1]:
                ret.append(1 + max([r for r, obs in zip(ret, obstacles) if obs <= obstacles[i]] or [0]))
            else:
                ret.append(1 + max([r for r, obs in zip(ret, obstacles) if obs <= obstacles[i]] + [ret[-1]]))
        return ret

    def longestObstacleCourseAtEachPosition_alsoslow(self, obstacles: List[int]) -> List[int]:
        ret = [1]
        longest_streaks = {obstacles[0]: 1}
        keys = [obstacles[0]]

        def floor_dict_get(key):
            if key in longest_streaks:
                floor_idx = bisect.bisect_left(keys, key)
            else:
                floor_idx = bisect.bisect_left(keys, key) - 1
            if floor_idx == -1:
                return 0
            return max(longest_streaks.get(keys[i], 0) for i in range(floor_idx+1))

        for i in range(1, len(obstacles)):
            if obstacles[i] < obstacles[i-1]:  # longest_streaks
                res = 1 + floor_dict_get(obstacles[i])
            else:
                res = 1 + max(floor_dict_get(obstacles[i]), ret[-1])
            longest_streaks[obstacles[i]] = res
            bisect.insort(keys, obstacles[i])
            ret.append(res)

        return ret

    def longestObstacleCourseAtEachPosition(self, nums: List[int]) -> List[int]:
        lis = []  # Longest increasing subsequence so far
        ret = []
        for i, x in enumerate(nums):
            if len(lis) == 0 or lis[-1] <= x:  # Append to LIS if new element is >= last element in LIS
                lis.append(x)
                ret.append(len(lis))
            else:
                idx = bisect.bisect_right(lis, x)  # Find the index of the smallest number > x
                lis[idx] = x  # Replace that number with x
                ret.append(idx + 1)
        return ret

    def test4(self):
        print_assert(self.longestObstacleCourseAtEachPosition([1, 2, 3, 2]), [1, 2, 3, 3])
        print_assert(self.longestObstacleCourseAtEachPosition([2, 2, 1]), [1, 2, 1])
        print_assert(self.longestObstacleCourseAtEachPosition([3, 1, 5, 6, 4, 2]), [1, 1, 2, 3, 2, 2])
        print_assert(self.longestObstacleCourseAtEachPosition([5, 1, 5, 5, 1, 3, 4, 5, 1, 4]),
                                                              [1, 1, 2, 3, 2, 3, 4, 5, 3, 5])
        print_assert(self.longestObstacleCourseAtEachPosition([5, 2, 5, 4, 1, 1, 1, 5, 3, 1]),
                                                              [1, 1, 2, 2, 1, 2, 3, 4, 4, 4])
        print_assert(self.longestObstacleCourseAtEachPosition([2, 2, 3, 5, 1, 4, 4, 1, 5, 1]),
                                                              [1, 2, 3, 4, 1, 4, 5, 2, 6, 3])

if __name__ == '__main__':
    sol = Solution()
    sol.test4()
    # def floor_key(d, key):
    #     if key in d:
    #         return key
    #     keys = sorted(d.keys())
    #     idx = bisect.bisect_left(keys, key) - 1
    #     return keys[idx]
    #
    # d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 6: 'f'}
    # print(floor_key(d, 1))
    # print(floor_key(d, 2))
    # print(floor_key(d, 3))
    # print(floor_key(d, 4))
    # print(floor_key(d, 5))
    # print(floor_key(d, 6))
    # print(floor_key(d, 7))

