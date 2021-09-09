from typing import List, Generator
from bisect import bisect_left, bisect_right
from helpers import print_assert


class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        c = 0
        for w in text.split():
            for bl in brokenLetters:
                if bl in w:
                    break
            else:
                c+= 1
        return c

    def addRungs(self, rungs: List[int], dist: int) -> int:
        c = 0
        # check first rung
        c += (rungs[0]-1) // dist
        for i in range(len(rungs)-1):
            c += (rungs[i+1]-rungs[i]-1) // dist
        return c

    def maxPoints(self, points: List[List[int]]) -> int:
        import heapq

        m, n = len(points), len(points[0])
        # construct m x n dp array, first row same as points
        dp = [points[0]] + [[0] * n for _ in range(m-1)]
        for r in range(1, m):
            # max1, max2, idx1, idx2 = self.max2([dp[r-1][x] for x in range(n)])
            # if max1 - max2 > n:
            #     for c in range(n):
            #         dp[r][c] = max1 - abs(c-idx1) + points[r][c]
            # else:
            for c in range(n):
                dp[r][c] = max(dp[r-1][x] - abs(c-x) for x in range(n)) + points[r][c]
        return max(dp[-1])
    #
    # def max2(self, l: List):
    #     if len(l) <= 1:
    #         return -1, -1, -1, -1
    #     if l[0] > l[1]:
    #         m1, m2 = l[0], l[1]
    #         i1, i2 = 0, 1
    #     else:
    #         m2, m1 = l[0], l[1]
    #         i2, i1 = 0, 1
    #     for i in range(2, len(l)):
    #         if l[i] > m1:
    #             m2 = m1
    #             i2 = i1
    #             m1 = l[i]
    #             i1 = i
    #         elif l[i] > m2:
    #             m2 = l[i]
    #             i2 = i
    #     return m1, m2, i1, i2
    # def argmax(self, l:List):
    #     i1 = 0
    #     m1 = l[0]
    #     for i in range(1, len(l)):
    #         if l[i] > m1:
    #             m1 = l[i]
    #             i1 = i
    #     return m1, i1
    #
    # def maxPoints(self, points: List[List[int]]) -> int:
    #     # THIS IS WRONG. STUCK IN LOCAL MINIMUM
    #     # initialize as the biggest element on each row
    #     # try to adjust each row to make result bigger
    #     chosen = []
    #     score = 0
    #     m, n = len(points), len(points[0])
    #     for r in range(m):
    #         max1, idx1= self.argmax(points[r])
    #         score += max1
    #         chosen.append(idx1)
    #     for i in range(m-1):
    #         score -= abs(chosen[i] - chosen[i+1])
    #
    #     prev_score = score
    #     while True:
    #         for r in range(m):
    #             for c in range(n):
    #                 score_increase = points[r][c] - points[r][chosen[r]]
    #                 penalty_increase = 0
    #                 if r > 0:
    #                     penalty_increase += abs(chosen[r-1] - c) - abs(chosen[r-1] - chosen[r])
    #                 if r < m-1:
    #                     penalty_increase += abs(chosen[r+1] - c) - abs(chosen[r+1] - chosen[r])
    #                 if score_increase - penalty_increase > 0:
    #                     score += score_increase - penalty_increase
    #                     chosen[r] = c
    #
    #         if score == prev_score:
    #             break
    #         else:
    #             prev_score = score
    #     return score


if __name__ == '__main__':
    sol = Solution()
    # print_assert(sol.canBeTypedWords('hello world', 'ad'), 1)
    # print_assert(sol.canBeTypedWords('leet code', 'lt'), 1)
    # print_assert(sol.canBeTypedWords('leet code', 'e'), 0)
    # print_assert(sol.canBeTypedWords('afaf asf asfs dfss w qawer', ''), 6)
    # print_assert(sol.canBeTypedWords('afaf asf asfs dfss w qawer', 'w'), 4)
    # print_assert(sol.canBeTypedWords('afaf asf asfs dfss w qawer', 'aw'), 1)

    # print_assert(sol.addRungs([1, 3, 5, 10], 2), 2)
    # print_assert(sol.addRungs([3, 6, 8, 10], 3), 0)
    # print_assert(sol.addRungs([3, 4, 6, 7], 2), 1)
    # print_assert(sol.addRungs([5], 10), 0)
    # print_assert(sol.addRungs([1, 2, 3, 4, 5], 10), 0)
    # print_assert(sol.addRungs([1, 5, 10, 30], 9), 2)

    print_assert(sol.maxPoints([[1,2,3],[1,5,1],[3,1,1]]), 9)
    print_assert(sol.maxPoints([[1,5],[2,3],[4,2]]), 11)
    print_assert(sol.maxPoints([[1, 2, 3],[5, 1, 1],[1, 1, 2]]), 7)
    print_assert(sol.maxPoints([[0,3,0,4,2],[5,4,2,4,1],[5,0,0,5,1],[2,0,1,0,3]]), 15)
    print_assert(sol.maxPoints([[5],[3],[2]]), 10)
    # print_assert(sol.max2([1, 2, 56, 9, 0, 3, 45]), (56, 45, 2, 6))