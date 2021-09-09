import heapq
import itertools
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Tuple, Optional

from helpers import print_assert, TreeNode, Tree as tr, Node, NaryTree as ntr


class Q76:
    # Minimum Window Substring
    # Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that
    # every character in t (including duplicates) is included in the window. If there is no such substring,
    # return the empty string "".
    def minWindow(self, s: str, t: str) -> str:
        m, n = len(s), len(t)
        t_counts = Counter(t)
        unfulfilled_chars = set(t_counts.keys())
        start, end = 0, 0
        min_start, min_end = 0, m + 1

        while end <= m:
            if len(unfulfilled_chars) == 0:  # all chars fulfilled, try to make substr shorter
                if end - start < min_end - min_start:
                    # update shortest length
                    min_start, min_end = start, end
                if s[start] in t_counts:
                    # check if a shorter substr makes it invalid
                    t_counts[s[start]] += 1
                    if t_counts[s[start]] > 0:
                        unfulfilled_chars.add(s[start])
                start += 1

            elif end < m:
                # some chars are not fulfilled
                if t_counts[s[end]] == 1:
                    unfulfilled_chars.remove(s[end])
                    t_counts[s[end]] -= 1
                elif s[end] in t_counts:
                    t_counts[s[end]] -= 1
                end += 1

            else:  # end == m:
                break

        if min_end - min_start == m + 1:  # it was never updated
            return ""
        return s[min_start:min_end]

    def test(self):
        print_assert(self.minWindow(s="ADOBECODEBANC", t="ABC"), "BANC")
        print_assert(self.minWindow(s="a", t="a"), "a")
        print_assert(self.minWindow(s="a", t="aa"), "")


class NumArray:
    # 303. Range Sum Query - Immutable
    # Given an integer array nums, handle multiple queries of the following type:
    #
    # Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
    def __init__(self, nums: List[int]):
        self.cumul_sum = [0]
        for num in nums:
            self.cumul_sum.append(self.cumul_sum[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        return self.cumul_sum[right + 1] - self.cumul_sum[left]


class Q303:
    def test(self):
        sol = NumArray([-2, 0, 3, -5, 2, -1])
        print_assert(sol.sumRange(0, 2), 1)
        print_assert(sol.sumRange(2, 5), -1)
        print_assert(sol.sumRange(0, 5), -3)


class Q265:
    def minCostII(self, costs: List[List[int]]) -> int:
        n, k = len(costs), len(costs[0])
        if n == 1:
            return min(costs[0])
        last_row = []
        for house in range(n):

            first_min = second_min = float('inf')
            first_min_idx = 0
            for i in range(k):
                cost = costs[house][i]
                if house > 0:
                    # min_prev_cost = min(prev_row[:i]+prev_row[i+1:])
                    min_prev_cost = prev_first_min if i != prev_first_min_idx else prev_second_min
                    cost += min_prev_cost
                    if house == n - 1:
                        last_row.append(cost)

                if cost < first_min:
                    second_min, first_min = first_min, cost
                    first_min_idx = i
                elif cost < second_min:
                    second_min = cost

            prev_first_min, prev_second_min, prev_first_min_idx = first_min, second_min, first_min_idx

        return min(last_row)

    # Time: O(NK). Space: O(K)

    def test(self):
        print_assert(self.minCostII([[1, 5, 3], [2, 9, 4]]), 5)
        print_assert(self.minCostII([[1, 3], [2, 4]]), 5)
        print_assert(self.minCostII([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]), 19)
        print_assert(self.minCostII([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 10, 13, 14, 15]]), 18)
        print_assert(self.minCostII([[1, 2]]), 1)


class Q1448:
    # 1448. Count Good Nodes in Binary Tree
    # Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with
    # a value greater than X.
    #
    # Return the number of good nodes in the binary tree.
    def goodNodes(self, root: TreeNode) -> int:
        return self.goodNodesHelper(root, -10 ** 4 - 1)

    def goodNodesHelper(self, root: TreeNode, biggest_above: int) -> int:
        if root is None:
            return 0

        biggest = max(root.val, biggest_above)
        return self.goodNodesHelper(root.left, biggest) + \
               self.goodNodesHelper(root.right, biggest) + \
               int(root.val >= biggest_above)
    # Time: O(N), Space: O(N)

    def test(self):
        null = None
        print_assert(self.goodNodes(tr.list2tree([3, 1, 4, 3, null, 1, 5])), 4)
        print_assert(self.goodNodes(tr.list2tree([3, 3, null, 4, 2])), 3)
        print_assert(self.goodNodes(tr.list2tree([1])), 1)


class Q91:
    # Decode Ways
    # To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of
    # the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
    #
    # "AAJF" with the grouping (1 1 10 6)
    # "KJF" with the grouping (11 10 6)
    # Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from
    # "06".
    #
    # Given a string s containing only digits, return the number of ways to decode it.
    @lru_cache(None)
    def numDecodings(self, s: str) -> int:
        # if len(s) <= 1:
        #     return len(s)
        # if int(s[:2]) <= 26:
        #     return self.numDecodings(s[2:]) + self.numDecodings(s[1:])
        # else:
        #     return self.numDecodings(s[1:])
        return self.numDecodingsHelper(s, start=0)

    @lru_cache(None)
    def numDecodingsHelper(self, s: str, start: int) -> int:
        if start < len(s) and s[start] == '0':
            return 0
        elif len(s) - start <= 1:
            return 1
        if int(s[start:start+2]) <= 26:
            return self.numDecodingsHelper(s, start+2) + self.numDecodingsHelper(s, start+1)
        else:
            return self.numDecodingsHelper(s, start+1)

    def test(self):
        print_assert(self.numDecodings("12"), 2)
        print_assert(self.numDecodings("226"), 3)
        print_assert(self.numDecodings("12"), 2)
        print_assert(self.numDecodings("0"), 0)
        print_assert(self.numDecodings("06"), 0)
        print_assert(self.numDecodings("99999"), 1)
        print_assert(self.numDecodings("11111"), 8)


class Q1339:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        all_sums = []

        def subtreeSums(root: Optional[TreeNode]) -> int:
            '''
            return a list of all the subtree sums from the current root, with the biggest (the sum including the root) being
            the last element of the list
            '''
            if root is None:
                return 0
            left_sums = subtreeSums(root.left)
            right_sums = subtreeSums(root.right)
            s = left_sums + right_sums + root.val
            all_sums.append(s)
            return s

        total_sum = subtreeSums(root)
        # x = max(sums, key=lambda x: (total_sum-x)*x )
        x = min(all_sums, key=lambda x: abs(total_sum-x-x))
        return x*(total_sum - x) % (10**9 + 7)

    def test(self):
        null = None
        print_assert(self.maxProduct(tr.list2tree([1,2,3,4,5,6])), 110)
        print_assert(self.maxProduct(tr.list2tree([1,null,2,3,4,null,null,5,6])), 90)
        print_assert(self.maxProduct(tr.list2tree([2,3,9,10,7,8,6,5,4,11,1])), 1025)
        print_assert(self.maxProduct(tr.list2tree([1,1])), 1)


if __name__ == '__main__':
    Q1339().test()
