import heapq
import itertools
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Tuple

from helpers import print_assert, TreeNode, Tree as tr, Node, NaryTree as ntr


class Q827:
    # 827. Making A Large Island
    # You are given an n x n binary matrix grid. You are allowed to change at most one 0 to be 1.
    #
    # Return the size of the largest island in grid after applying this operation.
    def largestIsland(self, grid: List[List[int]]) -> int:
        # first pass: find the islands in the existing grid
        n = len(grid)  # = len(grid[0])
        island_grid = [[None] * n for _ in range(n)]
        island_lookup = defaultdict(list)  # island_idx -> [(i1, j1), (i2, j2), ...]
        new_island_idx = 1  # avoid 0 for syntax reasons, since bool(0) and bool(None) are both False
        for i in range(n):
            for j in range(n):
                if grid[i][j]:  # need to register current land as part of an existing island or a new island
                    # check left and up.
                    # if left xor up is a known island, then attach current cell to that island
                    # if left and up are known islands but different names, then combine them to one island
                    left_cell = island_grid[i][j - 1] if j > 0 else None
                    up_cell = island_grid[i - 1][j] if i > 0 else None
                    if left_cell or up_cell:
                        # attach to existing island, possibly merging two differently named islands
                        if left_cell and up_cell and left_cell != up_cell:  # combine two
                            # change all occurrences of left_cell to up_cell
                            for (ix, jx) in island_lookup[left_cell]:
                                island_grid[ix][jx] = up_cell
                            island_lookup[up_cell].extend(island_lookup.pop(left_cell))
                            register_cur = up_cell
                        else:
                            register_cur = left_cell or up_cell  # whichever one is not None

                    else:  # register a new island
                        register_cur = new_island_idx
                        new_island_idx += 1

                    # register island
                    island_grid[i][j] = register_cur
                    island_lookup[register_cur].append((i, j))

        if len(island_lookup) == 0:
            return 1  # there are no islands on the map
        # second pass: find out which 0->1 flip results in largest island mergers
        max_island_size = max(len(cell_list) for cell_list in island_lookup.values())
        for i in range(n):
            for j in range(n):
                if not grid[i][j]:
                    # check left, up, right, down for islands, connect possible islands
                    mergeable = set([])
                    # left
                    if j > 0 and island_grid[i][j - 1]:
                        mergeable.add(island_grid[i][j - 1])
                    # up
                    if i > 0 and island_grid[i - 1][j]:
                        mergeable.add(island_grid[i - 1][j])
                    # right
                    if j < n - 1 and island_grid[i][j + 1]:
                        mergeable.add(island_grid[i][j + 1])
                    # down
                    if i < n - 1 and island_grid[i + 1][j]:
                        mergeable.add(island_grid[i + 1][j])
                    max_island_size = max(max_island_size, 1 + sum(len(island_lookup[m]) for m in mergeable))
        return max_island_size

    # Time O(N^2), Space O(N^2)

    def test(self):
        print_assert(actual=self.largestIsland([[1, 0], [0, 1]]), expected=3)
        print_assert(actual=self.largestIsland([[1, 1], [1, 0]]), expected=4)
        print_assert(actual=self.largestIsland([[1, 1], [1, 1]]), expected=4)
        print_assert(actual=self.largestIsland([[0, 0], [0, 0]]), expected=1)
        print_assert(actual=self.largestIsland([[1, 1], [0, 0]]), expected=3)
        print_assert(actual=self.largestIsland([[0, 0], [1, 1]]), expected=3)
        print_assert(actual=self.largestIsland([[0, 1], [0, 1]]), expected=3)
        print_assert(actual=self.largestIsland([[0, 0, 0, 0, 0, 1, 1, 1],
                                                [0, 1, 0, 0, 1, 0, 0, 1],
                                                [0, 1, 1, 1, 1, 0, 0, 1],
                                                [0, 0, 1, 0, 1, 1, 0, 0],
                                                [1, 0, 0, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 1, 0, 1],
                                                [0, 0, 0, 1, 0, 1, 1, 0]]), expected=15)


class Q1:
    # Given an array of integers nums and an integer target,
    # return indices of the two numbers such that they add up to target.
    def twoSum_sortedarr(self, nums: List[int], target: int) -> List[int]:
        import numpy as np
        idx_arr = np.argsort(nums)
        i2 = len(nums) - 1
        for i1 in range(len(nums)):
            while i2 > i1:
                s = nums[idx_arr[i1]] + nums[idx_arr[i2]]
                if s == target:
                    return [idx_arr[i1], idx_arr[i2]]
                elif s > target:
                    i2 -= 1
                else:  # s < target
                    break  # increase i1

    # Time O(N log(N)), Space O(log(N)) due to numpy's quicksort

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        wanted = {}  # wanted second half to make up target -> idx of first half
        for i in range(len(nums)):
            if nums[i] in wanted:
                return [wanted[nums[i]], i]
            else:
                wanted[target - nums[i]] = i

    # Time O(N), Space O(N) for dictionary

    def test(self):
        print_assert(actual=self.twoSum([2, 7, 11, 15], 9), expected=[0, 1])
        print_assert(actual=self.twoSum([3, 2, 4], 6), expected=[1, 2])
        print_assert(actual=self.twoSum([3, 3], 6), expected=[0, 1])


class Q1168:
    # There are n houses in a village. We want to supply water for all the houses by building wells and laying pipes.
    #
    # For each house i, we can either build a well inside it directly with cost wells[i - 1]
    # (note the -1 due to 0-indexing), or pipe in water from another well to it.
    # The costs to lay pipes between houses are given by the array pipes, where each
    # pipes[j] = [house1j, house2j, costj] represents the cost to connect house1j and house2j
    # together using a pipe. Connections are bidirectional.
    #
    # Return the minimum total cost to supply water to all houses.
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        # solution from leetcode
        # 1. build an adjacency list representation of the graph
        # bidirectional graph represented in adjacency list
        graph = defaultdict(list)

        # add a virtual vertex indexed with 0.
        #   then add an edge to each of the house weighted by the cost
        for index, cost in enumerate(wells):
            graph[0].append((cost, index + 1))

        # add the bidirectional edges to the graph
        for house_1, house_2, cost in pipes:
            graph[house_1].append((cost, house_2))
            graph[house_2].append((cost, house_1))

        # A set to maintain all the vertex that has been added to
        #   the final MST (Minimum Spanning Tree),
        #   starting from the vertex 0.
        mst_set = {0}

        # heap to maitain the order of edges to be visited,
        #   starting from the edges originated from the vertex 0.
        # Note: we can start arbitrarily from any node.
        heapq.heapify(graph[0])
        edges_heap = graph[0]

        total_cost = 0
        while len(mst_set) < n + 1:
            cost, next_house = heapq.heappop(edges_heap)
            if next_house not in mst_set:
                # adding the new vertex into the set
                mst_set.add(next_house)
                total_cost += cost
                # expanding the candidates of edge to choose from
                #   in the next round
                for new_cost, neighbor_house in graph[next_house]:
                    if neighbor_house not in mst_set:
                        heapq.heappush(edges_heap, (new_cost, neighbor_house))

        return total_cost


class Q90:
    # 90. Subsets II
    # Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        counts = Counter(nums).items()
        # for each item[i] and count[i], item[i] can be included 0, 1, 2, ... count[i] times in the result
        res = []
        keys, values = zip(*counts)
        for num_repeats in itertools.product(*[range(v + 1) for v in values]):
            subset = []
            for k, n in zip(keys, num_repeats):
                subset += [k] * n
            res.append(subset)

        return res

    def test(self):
        trials = [[1, 2, 2], [0], [2, 2, 2], [5, 4, 3, 2, 1]]
        for t in trials:
            res = self.subsetsWithDup(t)
            print(f"Power set of {t}:", res, "length", len(res))


class Q113:
    # 113. Path Sum II
    # Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where each path's
    # sum equals targetSum.
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        if root is None:
            return []
        if root.left is None and root.right is None:  # leaf node
            if targetSum == root.val:
                return [[root.val]]
            else:
                return []

        results = self.pathSum(root.left, targetSum - root.val) + self.pathSum(root.right, targetSum - root.val)
        for p in results:
            p.insert(0, root.val)
        return results

    # Time O(N), Space O(N)

    def test(self):
        null = None
        print_assert(actual=self.pathSum(tr.list2tree([5, 4, 8, 11, null, 13, 4, 7, 2, null, null, 5, 1]), 22),
                     expected=[[5, 4, 11, 2], [5, 8, 4, 5]])
        print_assert(actual=self.pathSum(tr.list2tree([1, 2, 3]), 5),
                     expected=[])
        print_assert(actual=self.pathSum(tr.list2tree([1, 2]), 0),
                     expected=[])
        print_assert(actual=self.pathSum(tr.list2tree([1, 2]), 3),
                     expected=[[1, 2]])
        print_assert(actual=self.pathSum(tr.list2tree([1]), 3),
                     expected=[])
        print_assert(actual=self.pathSum(tr.list2tree([]), 3),
                     expected=[])
        print_assert(actual=self.pathSum(tr.list2tree([1]), 1),
                     expected=[[1]])


class Q877:
    # Alex and Lee play a game with piles of stones.  There are an even number of piles arranged in a row,
    # and each pile has a positive integer number of stones piles[i].
    #
    # The objective of the game is to end with the most stones. The total number of stones is odd, so there are no ties.
    #
    # Alex and Lee take turns, with Alex starting first.  Each turn, a player takes the entire pile of stones from
    # either the beginning or the end of the row.  This continues until there are no more piles left, at which point
    # the person with the most stones wins.
    #
    # Assuming Alex and Lee play optimally, return True if and only if Alex wins the game.
    def stoneGame(self, piles: List[int]) -> bool:
        # return self.alexWinsRecur(piles, 0, 0)
        return self.alexWinsDP(piles)

    def alexWinsRecur(self, piles: List[int], alex_score: int, lee_score: int) -> bool:
        if len(piles) == 2:
            return alex_score + max(piles) > lee_score + min(piles)
        # alex L, lee L2
        cond1 = self.alexWinsRecur(piles[2:], alex_score=alex_score + piles[0], lee_score=lee_score + piles[1])
        # alex L, lee R
        cond2 = self.alexWinsRecur(piles[1:-1], alex_score=alex_score + piles[0], lee_score=lee_score + piles[-1])
        # alex R, lee L
        cond3 = self.alexWinsRecur(piles[1:-1], alex_score=alex_score + piles[-1], lee_score=lee_score + piles[0])
        # alex R, lee L
        cond4 = self.alexWinsRecur(piles[:-2], alex_score=alex_score + piles[-1], lee_score=lee_score + piles[-2])

        return (cond1 and cond2) or (cond3 and cond4)

    def alexWinsDP_dict(self, piles: List[int]) -> bool:
        n = len(piles)
        dp = defaultdict(int)
        for gap in range(1, n, 2):
            for i in range(n - gap):
                j = i + gap
                dp[i, j] = max(piles[i] + min(dp[i + 2, j], dp[i + 1, j - 1]),
                               piles[j] + min(dp[i + 1, j - 1], dp[i, j - 2]))
        print("alex's score is ", dp[0, n - 1])
        return dp[0, n - 1] > sum(piles) - dp[0, n - 1]  # alex's score > lee's score

    def alexWinsDP(self, piles: List[int]) -> bool:
        n = len(piles)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for gap in range(1, n, 2):
            for i in range(n - gap):
                j = i + gap
                dp[i][j] = max(piles[i] + min(dp[i + 2][j], dp[i + 1][j - 1]),
                               piles[j] + min(dp[i + 1][j - 1], dp[i][j - 2]))
        print("alex's score is ", dp[0][n - 1])
        return dp[0][n - 1] > sum(piles) - dp[0][n - 1]  # alex's score > lee's score

    def test(self):
        print_assert(actual=self.stoneGame([5, 3, 4, 5]), expected=True)
        print_assert(actual=self.stoneGame([7, 5, 2, 8, 3, 9]), expected=True)
        print_assert(actual=self.stoneGame([3, 10, 8, 4]), expected=True)


class Q429:
    # Given an n-ary tree, return the level order traversal of its nodes' values.
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if root is None:
            return []
        ret = []
        cur_traversal = []

        cur_row = [root]
        children = []

        while len(cur_row) > 0:
            for node in cur_row:
                cur_traversal.append(node.val)
                if node.children:
                    children.extend(node.children)
            ret.append(cur_traversal)
            cur_traversal = []
            cur_row = children
            children = []

        return ret

    def test(self):
        null = None
        print_assert(actual=self.levelOrder(ntr.list2tree([1, null, 3, 2, 4, null, 5, 6])),
                     expected=[[1], [3, 2, 4], [5, 6]])
        print_assert(actual=self.levelOrder(ntr.list2tree([1, null, 2, 3, 4, 5, null, null, 6, 7, null, 8, null, 9, 10,
                                                           null, null, 11, null, 12, null, 13, null, null, 14])),
                     expected=[[1], [2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13], [14]])
        print_assert(actual=self.levelOrder(ntr.list2tree([])), expected=[])
        print_assert(actual=self.levelOrder(ntr.list2tree([1])), expected=[[1]])
        print_assert(actual=self.levelOrder(ntr.list2tree([1, null, 2])), expected=[[1], [2]])
        print_assert(actual=self.levelOrder(ntr.list2tree([1, null, 2, 3])), expected=[[1], [2, 3]])
        print_assert(actual=self.levelOrder(ntr.list2tree([1, null, 2, null, 3])), expected=[[1], [2], [3]])


class Q1220:
    # 1220. Count Vowels Permutation
    # Given an integer n, your task is to count how many strings of length n can be formed under the following rules:
    #
    # Each character is a lower case vowel ('a', 'e', 'i', 'o', 'u')
    # Each vowel 'a' may only be followed by an 'e'.
    # Each vowel 'e' may only be followed by an 'a' or an 'i'.
    # Each vowel 'i' may not be followed by another 'i'.
    # Each vowel 'o' may only be followed by an 'i' or a 'u'.
    # Each vowel 'u' may only be followed by an 'a'.
    def countVowelPermutation(self, n: int) -> int:
        cur_row = [1] * 5  # length one, five strings
        row_count = 1
        while row_count < n:
            prev_row = cur_row[:]
            cur_row[0] = prev_row[1] + prev_row[2] + prev_row[4]  # ea, ia, ua
            cur_row[1] = prev_row[0] + prev_row[2]  # ae, ie
            cur_row[2] = prev_row[1] + prev_row[3]  # ei, oi
            cur_row[3] = prev_row[2]  # io
            cur_row[4] = prev_row[2] + prev_row[3]  # iu, ou
            row_count += 1

        return sum(cur_row) % (10 ** 9 + 7)

    def test(self):
        print_assert(self.countVowelPermutation(1), 5)
        print_assert(self.countVowelPermutation(2), 10)
        print_assert(self.countVowelPermutation(5), 68)


class Q132:
    # Palindrome Partitioning II
    # Given a string s, partition s such that every substring of the partition is a palindrome.

    def minCut_slow(self, s: str) -> int:
        n = len(s)

        @lru_cache(None)
        def is_palindrome(substr):
            return all(substr[i] == substr[-1 - i] for i in range(len(substr) // 2))

        @lru_cache(None)
        def dp(pos):
            if is_palindrome(s[pos:]):
                return 0
            return 1 + min(dp(end) for end in range(pos+1, n) if is_palindrome(s[pos:end]))

        return dp(0)

    def minCut(self, s: str) -> int:
        cut = [-1] + [x for x in range(len(s))]

        for i in range(2 * len(s) - 1):
            left = i // 2
            right = i - left
            while right < len(s) and left >= 0 and s[right] == s[left]:
                cut[right + 1] = min(cut[right + 1], cut[left] + 1)
                right += 1
                left -= 1

        return cut[-1]

    def test(self):
        print_assert(self.minCut("aab"), 1)
        print_assert(self.minCut("a"), 0)
        print_assert(self.minCut("ab"), 1)
        print_assert(self.minCut("abababababa"), 0)
        print_assert(self.minCut("ababbababa"), 2)
        print_assert(self.minCut("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), 1)


if __name__ == '__main__':
    Q132().test()
