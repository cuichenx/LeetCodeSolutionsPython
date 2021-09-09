from typing import List

from helpers import print_assert
from bisect import insort_left, insort_right, bisect_left, bisect_right


class Q542:
    # 542. 01 Matrix
    # Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
    #
    # The distance between two adjacent cells is 1.
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        coord2dist = {}  # (i, j) -> distance to nearest 0
        m, n = len(mat), len(mat[0])
        cur_set = set([])
        cur_d = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    coord2dist[(i, j)] = 0
                    cur_set.add((i, j))

        while len(cur_set) > 0:
            cur_d += 1
            next_set = set([])
            for (i, j) in cur_set:
                if i > 0 and (i - 1, j) not in coord2dist:
                    coord2dist[(i - 1, j)] = cur_d
                    next_set.add((i - 1, j))
                if i < m - 1 and (i + 1, j) not in coord2dist:
                    coord2dist[(i + 1, j)] = cur_d
                    next_set.add((i + 1, j))
                if j > 0 and (i, j - 1) not in coord2dist:
                    coord2dist[(i, j - 1)] = cur_d
                    next_set.add((i, j - 1))
                if j < n - 1 and (i, j + 1) not in coord2dist:
                    coord2dist[(i, j + 1)] = cur_d
                    next_set.add((i, j + 1))
            cur_set = next_set

        ret = [[0] * n for _ in range(m)]
        for (i, j), d in coord2dist.items():
            ret[i][j] = d
        return ret

    # Time: O(mn), Space: O(mn)

    def test(self):
        print_assert(self.updateMatrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        print_assert(self.updateMatrix([[0, 0, 0], [0, 1, 0], [1, 1, 1]]), [[0, 0, 0], [0, 1, 0], [1, 2, 1]])
        print_assert(self.updateMatrix([[1, 1, 1, 0, 0, 1, 1, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 0, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 0, 1, 1, 1],
                                        [1, 1, 1, 1, 0, 1, 1, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1]]),
                     [[3, 2, 1, 0, 0, 1, 1, 0],
                      [4, 3, 2, 1, 1, 2, 2, 1],
                      [3, 2, 1, 2, 2, 3, 3, 2],
                      [2, 1, 0, 1, 2, 3, 4, 3],
                      [3, 2, 1, 2, 1, 2, 3, 2],
                      [4, 3, 2, 1, 0, 1, 2, 1],
                      [4, 3, 2, 1, 0, 1, 1, 0],
                      [5, 4, 3, 2, 1, 2, 2, 1]])


class Q677:
    # Map Sum Pairs
    # MapSum() Initializes the MapSum object.
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.words = []  # sorted
        self.values = []

    # void insert(String key, int val) Inserts the key-val pair into the map. If the key already existed,
    # the original key-value pair will be overridden to the new one.
    def insert(self, key: str, val: int) -> None:
        idx = bisect_left(self.words, key)
        if idx < len(self.words) and self.words[idx] == key:
            self.values[idx] = val  # overwrite with new val
        else:
            self.words.insert(idx, key)
            self.values.insert(idx, val)

    # Time: O(N), Space: O(1)

    # int sum(string prefix) Returns the sum of all the pairs' value whose key starts with the prefix.
    def sum(self, prefix: str) -> int:
        idx = bisect_left(self.words, prefix)
        s = 0
        while idx < len(self.words):
            if self.words[idx].startswith(prefix):
                s += self.values[idx]
                idx += 1
            else:
                break

        return s

    # Time: O(max(logN, k)) where k is the number of matches. Space: O(1)

    def test(self):
        self.insert("apple", 3)
        print_assert(self.sum("ap"), 3)
        self.insert("app", 2)
        print_assert(self.sum("ap"), 5)
        self.insert("apps", 4)
        self.insert("apply", 9)
        self.insert("somethingelse", 1000)
        print_assert(self.sum("app"), 18)
        print_assert(self.sum("appl"), 12)


class TrieNode:
    def __init__(self, score=0):
        self.score = score
        self.children = {}  # char -> TrieNode


class Q677_Trie:
    # Map Sum Pairs
    # MapSum() Initializes the MapSum object.
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        self.w2score = {}

    # void insert(String key, int val) Inserts the key-val pair into the map. If the key already existed,
    # the original key-value pair will be overridden to the new one.
    def insert(self, key: str, val: int) -> None:
        score_delta = val - self.w2score.get(key, 0)
        self.w2score[key] = val  # for next time
        # insert node
        cur = self.root
        for char in key:
            if char not in cur.children:
                cur.children[char] = TrieNode()  # add new branch with score only val
            cur = cur.children[char]  # go down a branch
            cur.score += score_delta

    # Time: O(K), K is length of key, Space: O(1)

    # int sum(string prefix) Returns the sum of all the pairs' value whose key starts with the prefix.
    def sum(self, prefix: str) -> int:
        cur = self.root
        for char in prefix:
            if char not in cur.children:
                return 0
            cur = cur.children[char]  # go down a branch
        return cur.score

    # Time: O(K), Space: O(1)

    def test(self):
        self.insert("apple", 3)
        print_assert(self.sum("ap"), 3)
        self.insert("app", 2)
        print_assert(self.sum("ap"), 5)
        self.insert("apps", 4)
        self.insert("apply", 9)
        self.insert("somethingelse", 1000)
        print_assert(self.sum("app"), 18)
        print_assert(self.sum("appl"), 12)


class Q42:
    # Trapping Rain Water
    # Given n non-negative integers representing an elevation map where the width of each bar is 1,
    # compute how much water it can trap after raining.
    def trap(self, height: List[int]) -> int:
        # going from left
        cumul_l = []
        cur_max_l = 0
        for h in height:
            if h > cur_max_l:
                cur_max_l = h
            cumul_l.append(cur_max_l)

        # going from right
        cumul_r = []  # remember this is reversed
        cur_max_r = 0
        for h in reversed(height):
            if h > cur_max_r:
                cur_max_r = h
            cumul_r.append(cur_max_r)

        water = 0
        for leftwall, rightwall, h in zip(cumul_l, reversed(cumul_r), height):
            water += min(leftwall, rightwall) - h
        return water
    # Time: O(N), Space O(N)

    def test(self):
        print_assert(self.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2]), 6)
        print_assert(self.trap([4, 2, 0, 3, 2, 5]), 9)


if __name__ == '__main__':
    Q42().test()
