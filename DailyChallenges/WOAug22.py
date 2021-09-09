import bisect
import heapq
import itertools
from collections import defaultdict, Counter, deque
from functools import lru_cache
from typing import List, Tuple, Optional
import math
from helpers import print_assert, TreeNode, Tree as tr, Node, NaryTree as ntr


class Q653:
    # 653. Two Sum IV - Input is a BST
    # Given the root of a Binary Search Tree and a target number k, return true if there exist two elements in the BST
    # such that their sum is equal to the given target.
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        # traverse through tree (BFS or DFS), adding the complement of each element to a hashmap/set
        # Lets do BFS
        q = deque([root])
        wanted = set([])
        while len(q) > 0:
            cur_node = q.popleft()
            if cur_node:
                if cur_node.val in wanted:
                    return True
                wanted.add(k - cur_node.val)
                q.append(cur_node.left)
                q.append(cur_node.right)
        return False

    def test(self):
        null = None
        print_assert(self.findTarget(tr.list2tree([5, 3, 6, 2, 4, null, 7]), 9), True)
        print_assert(self.findTarget(tr.list2tree([5, 3, 6, 2, 4, null, 7]), 28), False)
        print_assert(self.findTarget(tr.list2tree([2, 1, 3]), 4), True)
        print_assert(self.findTarget(tr.list2tree([2, 1, 3]), 1), False)
        print_assert(self.findTarget(tr.list2tree([2, 1, 3]), 3), True)


class Q261:
    # 261. Graph Valid Tree
    # You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges where
    # edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.
    #
    # Return true if the edges of the given graph make up a valid tree, and false otherwise.
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # just do a DFS. if you ever visit a node that has already been visited, return False. Else return True
        edges_dict = defaultdict(list)
        for a, b in edges:
            edges_dict[a].append(b)
            edges_dict[b].append(a)

        visited = set([])
        stack = deque([(0, -1)])  # (this_node, from_node)
        while len(stack) > 0:
            cur_node, from_node = stack.pop()
            if cur_node in visited:
                return False
            visited.add(cur_node)
            for neighbour in edges_dict[cur_node]:
                if neighbour != from_node:
                    stack.append((neighbour, cur_node))
        return len(visited) == n

    def test(self):
        print_assert(self.validTree(5, [[0, 1], [0, 2], [0, 3], [1, 4]]), True)
        print_assert(self.validTree(5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]), False)
        print_assert(self.validTree(5, [[0, 1], [1, 2], [3, 4]]), False)
        print_assert(self.validTree(2, []), False)


class Q537:
    # 537. Complex Number Multiplication
    # Given two complex numbers num1 and num2 as strings, return a string of the complex number that represents their
    # multiplications.
    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        num1_parts = num1.split("+")
        n1r, n1i = int(num1_parts[0]), int(num1_parts[1][:-1])
        num2_parts = num2.split("+")
        n2r, n2i = int(num2_parts[0]), int(num2_parts[1][:-1])
        resr, resi = n1r * n2r - n1i * n2i, n1r * n2i + n1i * n2r
        return f"{resr}+{resi}i"

    def test(self):
        print_assert(self.complexNumberMultiply("1+1i", "1+1i"), "0+2i")
        print_assert(self.complexNumberMultiply("1+-1i", "1+-1i"), "0+-2i")


class Q36:
    # 36. Valid Sudoku
    # Determine if a 9 x 9 Sudoku board is valid.
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        return all(
            self.isValidSeq(board, 'row', i) and self.isValidSeq(board, 'col', i) and self.isValidSeq(board, 'box', i)
            for i in range(9)
        )

    def isValidSeq(self, board: List[List[str]], seq_type: str, seq_num: int):
        """
        seq_type: one of 'row', 'col', 'box'
        seq_num: 0-8, index of row, col, or box
        """
        cnter = [False] * 10

        def isNotRepeated(entry):
            if entry == '.':
                return True
            elif cnter[int(entry)]:
                return False
            else:
                cnter[int(entry)] = True
                return True

        if seq_type == 'row':
            return all(isNotRepeated(board[seq_num][j]) for j in range(9))
        elif seq_type == 'col':
            return all(isNotRepeated(board[i][seq_num]) for i in range(9))
        elif seq_type == 'box':
            return all(isNotRepeated(board[seq_num // 3 * 3 + i // 3][(seq_num % 3) * 3 + i % 3]) for i in range(9))

    def test(self):
        print_assert(self.isValidSudoku([["5", "3", ".", ".", "7", ".", ".", ".", "."]
                                            , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
                                            , [".", "9", "8", ".", ".", ".", ".", "6", "."]
                                            , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
                                            , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
                                            , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
                                            , [".", "6", ".", ".", ".", ".", "2", "8", "."]
                                            , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
                                            , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]), True)
        print_assert(self.isValidSudoku([["8", "3", ".", ".", "7", ".", ".", ".", "."]
                                            , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
                                            , [".", "9", "8", ".", ".", ".", ".", "6", "."]
                                            , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
                                            , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
                                            , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
                                            , [".", "6", ".", ".", ".", ".", "2", "8", "."]
                                            , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
                                            , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]), False)


class Q633:
    # Sum of Square Numbers
    # Given a non-negative integer c, decide whether there're two integers a and b such that a2 + b2 = c.
    def judgeSquareSum(self, c: int) -> bool:
        a = 0
        b = math.floor(c ** 0.5)
        while a <= b:
            s = a**2 + b**2
            if s == c:
                return True
            elif s < c:  # too small
                a += 1
            else:  # too big
                b -= 1

        return False

    def test(self):
        print_assert(self.judgeSquareSum(5), True)
        print_assert(self.judgeSquareSum(4), True)
        print_assert(self.judgeSquareSum(3), False)
        print_assert(self.judgeSquareSum(2), True)
        print_assert(self.judgeSquareSum(1), True)
        print_assert(self.judgeSquareSum(22**2+23**2), True)
        print_assert(self.judgeSquareSum(22**2+23**2+1), False)


class Q331:
    # Verify Preorder Serialization of a Binary Tree
    # Given a string of comma-separated values preorder, return true if it is a correct preorder traversal serialization
    # of a binary tree.
    def isValidSerialization(self, preorder: str) -> bool:
        # idea: keep a "stack counter".
        # when a value node is encountered, add to the stack with count 0
        # when a sentinel is encountered, increase the stack top by 1
        # if the counter reaches 2, pop the count and carry over 1
        # return true if the stack is empty in the end
        # if preorder[-3:] != '#,#':
        #     return False  # last two have to be sentinels
        symbols = preorder.split(',')
        if symbols[0] == '#':
            return preorder == '#'  # if first symbol is #, then this has to be an empty tree
        stack_counter = [0]
        for e in symbols[1:]:
            if len(stack_counter) == 0:
                return False
            if e != '#':
                stack_counter.append(0)  # we don't actually need to record the value of the node
            else:
                stack_counter[-1] += 1
            while stack_counter[-1] == 2:  # carry over
                stack_counter.pop(-1)
                if len(stack_counter) == 0:
                    break
                stack_counter[-1] += 1
        return len(stack_counter) == 0

    def test(self):
        print_assert(self.isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#"), True)
        print_assert(self.isValidSerialization("1,#"), False)
        print_assert(self.isValidSerialization("9,#,#,1"), False)
        print_assert(self.isValidSerialization("1,#,#,#,#"), False)
        print_assert(self.isValidSerialization("#"), True)
        print_assert(self.isValidSerialization("9,3,4,#,#,1,#,#,#,2,#,6,#,#"), False)
        print_assert(self.isValidSerialization("#,#,#"), False)


class Q1235:
    # 1235. Maximum Profit in Job Scheduling
    # We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of
    # profit[i].
    #
    # You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are
    # no two jobs in the subset with overlapping time range.
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        n = len(profit)
        start_time, end_time, profit = zip(*sorted(zip(startTime, endTime, profit)))

        @lru_cache(None)
        def dp(i):
            if i >= n:
                return 0  # if start after the last job started, then no hope
            # max(skip this job, take this job and dp[earliest job idx after end_time[i]])
            return max(dp(i+1), profit[i] + dp(bisect.bisect_left(start_time, end_time[i])))

        return dp(0)
    # Time: O(N logN). Space: O(N)

    def jobScheduling_bottomup(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        n = len(profit)
        start_time, end_time, profit = zip(*sorted(zip(startTime, endTime, profit)))
        t2soonest_idx = []
        i = 0
        for t in range(start_time[-1]+1):
            while start_time[i] < t:
                i += 1
            t2soonest_idx.append(i)

        dp = {}
        for t in reversed(sorted(end_time)):
            if t > start_time[-1]:
                dp[t] = 0
            else:
                dp[t] = max(profit[i] + dp[end_time[i]] for i in range(t2soonest_idx[t], n))

        # return dp[0]
        # THIS RECURRENCE ALWAYS TLE
        return max(profit[i] + dp[end_time[i]] for i in range(0, n))
    # Time: O(N logN). Space: O(N)

    def test(self):
        print_assert(self.jobScheduling([1, 2, 3, 3], [3, 4, 5, 6], [50, 10, 40, 70]), 120)
        print_assert(self.jobScheduling([1, 2, 3, 4, 6], [3, 5, 10, 6, 9], [20, 20, 100, 70, 60]), 150)
        print_assert(self.jobScheduling([1, 1, 1], [2, 3, 4], [5, 6, 4]), 6)
        print_assert(self.jobScheduling([4, 2, 4, 8, 2], [5, 5, 5, 10, 8], [1, 2, 8, 10, 4]), 18)


class Q330:
    # 330. Patching Array
    # Given a sorted integer array nums and an integer n, add/patch elements to the array such that any number in the
    # range [1, n] inclusive can be formed by the sum of some elements in the array.
    #
    # Return the minimum number of patches required.
    def minPatches(self, nums: List[int], n: int) -> int:
        # from https://leetcode.com/problems/patching-array/discuss/338621/Python-O(n)-with-detailed-explanation
        covered=0
        res=0
        for num in nums:
            while num>covered+1:
                res+=1
                covered=covered*2+1
                if covered>=n:
                    return res
            covered+=num
            if covered>=n:
                return res
        while covered<n:
            res+=1
            covered=covered*2+1
        return res

if __name__ == '__main__':
    Q1235().test()
