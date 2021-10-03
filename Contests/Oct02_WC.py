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
    def minimumMoves(self, s: str) -> int:
        i = 0
        num_moves = 0
        while i < len(s):
            if s[i] == 'O':
                i += 1
            else:
                num_moves += 1
                i += 3
        return num_moves

    def test1(self):
        print_assert(self.minimumMoves('XXX'), 1)
        print_assert(self.minimumMoves('XXOX'), 2)
        print_assert(self.minimumMoves('OOOO'), 0)

    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        tot_sum = mean * (len(rolls) + n)
        avg_roll = (tot_sum - sum(rolls))/n
        if avg_roll < 1 or avg_roll > 6:
            return []
        res = []
        num_lower = int((math.ceil(avg_roll) - avg_roll) * n + 0.5)
        for i in range(n):
            res.append(math.floor(avg_roll) if i < num_lower else math.ceil(avg_roll))
        return res

    def test2(self):
        print_assert(self.missingRolls([3, 2, 4, 3], 4, 2), [6, 6])
        print_assert(self.missingRolls([1, 5, 6], 3, 4), [2, 2, 2, 3])
        print_assert(self.missingRolls([1, 2, 3, 4], 6, 4), [])
        print_assert(self.missingRolls([1], 3, 1), [5])
        print_assert(self.missingRolls([4,5,6,2,3,6,5,4,6,4,5,1,6,3,1,4,5,5,3,2,3,5,3,2,1,5,4,3,5,1,5], 4, 40),
                     [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5])

    def stoneGameIX(self, stones: List[int]) -> bool:
        # return whether alice can win
        @lru_cache(None)
        def can_win(cur_sum, rem0, rem1, rem2, is_alice):
            if rem0 + rem1 + rem2 == 0:
                return not is_alice
            if cur_sum == 0:
                # first one only
                win_if_take_1 = not can_win(1, rem0, rem1-1, rem2, not is_alice) if rem1 > 0 else False
                win_if_take_2 = not can_win(2, rem0, rem1, rem2-1, not is_alice) if rem2 > 0 else False
                return win_if_take_1 or win_if_take_2
            elif cur_sum == 1:
                win_if_take_0 = not can_win(1, rem0-1, rem1, rem2, not is_alice) if rem0 > 0 else False
                win_if_take_1 = not can_win(2, rem0, rem1-1, rem2, not is_alice) if rem1 > 0 else False
                return win_if_take_0 or win_if_take_1
            else:  # cur_sum == 2:
                win_if_take_0 = not can_win(2, rem0-1, rem1, rem2, not is_alice) if rem0 > 0 else False
                win_if_take_2 = not can_win(1, rem0, rem1, rem2-1, not is_alice) if rem2 > 0 else False
                return win_if_take_0 or win_if_take_2

        remainder_count = [0, 0, 0]
        for c in stones:
            remainder_count[c%3] += 1
        if min(remainder_count) > 2:
            sub = (min(remainder_count)//2-1)*2
        else:
            sub = 0

        return can_win(0, remainder_count[0]-sub, remainder_count[1]-sub, remainder_count[2]-sub, True)

    def test3(self):
        print_assert(self.stoneGameIX([2, 1]), True)
        print_assert(self.stoneGameIX([2]), False)
        print_assert(self.stoneGameIX([5, 1, 2, 4, 3]), False)
        print_assert(self.stoneGameIX([5, 1, 2, 4, 3, 6]), True)
        print_assert(self.stoneGameIX([77,74,12,63,95,23,19,91,48,87,26,22,21,30,41,10,22,80,14,36,62,29,13,3,15,47,71,1,95,21,43,84,62,70,10,86,70,9,38,30,51,32,75,87,73,8,54,64,35,22,68,75,4,59,69,82,27,9,20,32,64,59,58,48,32,21,15,20,75]), True)
        print_assert(self.stoneGameIX([0]*7+[2]*8), True)
        print_assert(self.stoneGameIX([0]*9 + [1]*2 + [2]*10), True)
        print_assert(self.stoneGameIX([0]*11 + [1]*4 + [2]*12), True)
        print_assert(self.stoneGameIX([0]*13 + [1]*6 + [2]*14), True)
        print_assert(self.stoneGameIX([0]*15 + [1]*8 + [2]*16), True)
        print_assert(self.stoneGameIX([0]*17 + [1]*10 + [2]*18), True)
        print_assert(self.stoneGameIX([0]*19 + [1]*12 + [2]*20), True)
        print_assert(self.stoneGameIX([0]*8 + [1]*10 + [2]*6), True)

    def smallestSubsequence(self, s: str, k: int, letter: str, repetition: int) -> str:
        global smallest_str
        smallest_str = 'z' * k

        @lru_cache(None)
        def recursion(start_idx, k, letter, repetition, cur_str):
            global smallest_str
            if start_idx >= len(s):
                if k == 0 and repetition <= 0:
                    # update answer
                    smallest_str = min(smallest_str, cur_str)
                    return
                return
            if cur_str > smallest_str:
                return
            # use cur letter
            recursion(start_idx+1, k-1, letter, repetition-int(s[start_idx] == letter), cur_str+s[start_idx])
            # dont use cur letter
            recursion(start_idx+1, k, letter, repetition, cur_str)

        recursion(0, k, letter, repetition, '')

        return smallest_str

    def test4(self):
        print_assert(self.smallestSubsequence('leetcode', 4, 'e', 2), 'ecde')
        # print_assert(self.smallestSubsequence('leet', 3, 'e', 1), 'eet')

if __name__ == '__main__':
    Solution().test4()


