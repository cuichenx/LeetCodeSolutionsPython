from typing import List

from helpers import print_assert
import heapq
import numpy as np

class Solution:
    def makeFancyString(self, s: str) -> str:
        i = 0
        while i < len(s) - 2:
            if s[i] == s[i+1] == s[i+2]:
                s = s[:i] + s[i+1:]
            else:
                i += 1
        return s

    def test1(self):
        print_assert(self.makeFancyString('leeetcode'), 'leetcode')
        print_assert(self.makeFancyString('leeeetcode'), 'leetcode')
        print_assert(self.makeFancyString('leeeeetcode'), 'leetcode')
        print_assert(self.makeFancyString('leeeeetcooode'), 'leetcoode')
        print_assert(self.makeFancyString('leeeeetcooode'), 'leetcoode')
        print_assert(self.makeFancyString('aaabaaaa'), 'aabaa')
        print_assert(self.makeFancyString('a'), 'a')
        print_assert(self.makeFancyString('aa'), 'aa')
        print_assert(self.makeFancyString('aaa'), 'aa')
        print_assert(self.makeFancyString('aab'), 'aab')

    def checkMove(self, board: List[List[str]], rMove: int, cMove: int, color: str) -> bool:
        def check(rDelta, cDelta):
            try:
                if board[rMove + rDelta][cMove + cDelta] == ('B' if color == 'W' else 'W'):  # opposite colour
                    step = 2
                    while 0 <= rMove + step * rDelta < 8 and 0 <= cMove + step * cDelta < 8:
                        val = board[rMove + step*rDelta][cMove + step*cDelta]
                        if val == color:
                            return True
                        elif val == '.':
                            break
                        step += 1
            except IndexError:
                pass
            return False

        return any([check(-1, 0), check(1, 0), check(0, -1), check(0, 1),  # up, down, left, right
                    check(-1, -1), check(-1, 1), check(1, -1), check(1, 1)])  # upleft, upright, downleft, downright

    def test2(self):
        print_assert(self.checkMove(
            board=[[".", ".", ".", "B", ".", ".", ".", "."],
                   [".", ".", ".", "W", ".", ".", ".", "."],
                   [".", ".", ".", "W", ".", ".", ".", "."],
                   [".", ".", ".", "W", ".", ".", ".", "."],
                   ["W", "B", "B", ".", "W", "W", "W", "B"],
                   [".", ".", ".", "B", ".", ".", ".", "."],
                   [".", ".", ".", "B", ".", ".", ".", "."],
                   [".", ".", ".", "W", ".", ".", ".", "."]],
            rMove=4, cMove=3, color="B"), True)
        print_assert(self.checkMove(
            board=[[".", ".", ".", ".", ".", ".", ".", "."],
                   [".", "B", ".", ".", "W", ".", ".", "."],
                   [".", ".", "W", ".", ".", ".", ".", "."],
                   [".", ".", ".", "W", "B", ".", ".", "."],
                   [".", ".", ".", ".", ".", ".", ".", "."],
                   [".", ".", ".", ".", "B", "W", ".", "."],
                   [".", ".", ".", ".", ".", ".", "W", "."],
                   [".", ".", ".", ".", ".", ".", ".", "B"]],
            rMove=4, cMove=4, color="W"), False)
        print_assert(self.checkMove(
            board=[[".", ".", ".", ".", ".", ".", ".", "."],
                   [".", "B", ".", ".", "W", ".", ".", "."],
                   [".", ".", "W", ".", ".", ".", ".", "."],
                   [".", ".", ".", "W", "B", ".", ".", "."],
                   [".", ".", ".", ".", ".", ".", ".", "."],
                   [".", ".", ".", ".", "B", "W", ".", "."],
                   [".", ".", ".", ".", ".", ".", ".", "."],
                   [".", ".", ".", ".", ".", ".", ".", "B"]],
            rMove=4, cMove=4, color="B"), True)
        print_assert(self.checkMove(
            board=[["W", "W", ".", "B", ".", "B", "B", "."],
                   ["W", "B", ".", ".", "W", "B", ".", "."],
                   ["B", "B", "B", "B", "W", "W", "B", "."],
                   ["W", "B", ".", ".", "B", "B", "B", "."],
                   ["W", "W", "B", ".", "W", ".", "B", "B"],
                   ["B", ".", "B", "W", ".", "B", ".", "."],
                   [".", "B", "B", "W", "B", "B", ".", "."],
                   ["B", "B", "W", ".", ".", "B", ".", "."]],
            rMove=7, cMove=4, color="B"), True)

    def minSpaceWastedKResizing(self, nums: List[int], k: int) -> int:
        pq = [(0, nums)]
        cumul_savings = 0
        for i in range(k+1):  # first iter is trivial
            # pop the most valuable squeeze
            cur_saving, cur_list = heapq.heappop(pq)
            max_idx = max(range(len(cur_list)), key=lambda i: cur_list[i])
            cumul_savings -= cur_saving  # cur_saving is always negative due to python min heap

            # push the gain from this op and the sublist
            heapq.heappush(pq, (-self.squeezeDown(cur_list[:max_idx], cur_list[max_idx]), cur_list[:max_idx]))
            heapq.heappush(pq, (-self.squeezeDown(cur_list[max_idx+1:], cur_list[max_idx]), cur_list[max_idx+1:]))

        # space waster is max space - savings - actual space
        return max(nums) * len(nums) - cumul_savings - sum(nums)

    def decompose(self, nums: List[int]):
        # how much this segment can be pushed down and decomposed into subproblems
        # a segment can be decomposed in three ways:
        # 1. push the upper limit down to max(nums), giving a savings of (curLim - max(nums))*len(nums)
        # 2. starting from the leftmost bar, try to fit the biggest rectangle
        # 3. starting from the rightmost bar, try to fit the biggest rectangle
        if len(nums) == 0:
            return 0
        ...

    def squeezeDown(self, nums: List[int], curLim: int) -> int:
        if len(nums) > 0:
            return (curLim - max(nums))*len(nums)
        else:
            return 0

    def test3(self):
        print_assert(self.minSpaceWastedKResizing([10, 20, 15, 30, 20], 2), 15)
        print_assert(self.minSpaceWastedKResizing([10, 20], 0), 10)
        print_assert(self.minSpaceWastedKResizing([10, 20, 30], 1), 10)
        print_assert(self.minSpaceWastedKResizing([29, 38, 18, 1, 49, 11, 45, 28], 3), 68)

if __name__ == '__main__':
    sol = Solution()
    sol.test3()

