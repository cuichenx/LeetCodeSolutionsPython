import sys
from collections import deque
from functools import lru_cache
from typing import List, Optional, Dict

sys.path.append("..")
from helpers import print_assert, print_matrix, ListNode, LinkedList as ll
import copy
import time

FREE = 0
ATTACKABLE = 1
OCCUPIED = 2
class Q52:
    # 52. N-Queens II
    # Given an integer n, return the number of distinct solutions to the n-queens puzzle.

    def totalNQueens(self, n: int) -> int:
        self.n = n
        # 0 means available, 1 means occupied or attackable
        empty_board = [[FREE for _ in range(n)] for _ in range(n)]
        return sum(self.numQueensPartialBoard(0, j, copy.deepcopy(empty_board)) for j in range(n))  # can only search half the board

    def numQueensPartialBoard(self, i, j, board):
        board[i][j] = OCCUPIED
        if i == self.n - 1:
            # last row
            # print_matrix(board)
            # print('-' * 10)
            return 1

        # set new attackable cells
        self.setAttackableCells(i, j, board)

        # search for each row beneath it
        num_sol = 0
        for next_j in range(self.n):
            if board[i+1][next_j] == FREE:
                num_sol += self.numQueensPartialBoard(i+1, next_j, copy.deepcopy(board))
        return num_sol


    def setAttackableCells(self, i, j, board):
        # because we're searching row by row, there's no need to set the row as ATTACKABLE.
        # this column
        for col in range(i+1, self.n):
            board[col][j] = ATTACKABLE

        # this diagonal
        r, c = i+1, j+1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c += 1

        # this anti diagonal
        r, c = i+1, j-1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c -= 1

    def inRange(self, r, c):
        return (0 <= r < self.n) and (0 <= c < self.n)

    def test(self):
        print_assert(self.totalNQueens(1), 1)
        print_assert(self.totalNQueens(2), 0)
        print_assert(self.totalNQueens(3), 0)
        print_assert(self.totalNQueens(4), 2)
        print_assert(self.totalNQueens(5), 10)
        t0 = time.time()
        self.totalNQueens(9)
        print(time.time() - t0, 'seconds')

class Q51:
    # 52. N-Queens
    # Given an integer n, return all distinct solutions to the n-queens puzzle

    def solveNQueens(self, n: int) -> List[List[str]]:
        self.n = n
        # 0 means available, 1 means occupied or attackable
        empty_board = [[FREE for _ in range(n)] for _ in range(n)]
        self.solutions = []
        for j in range(n):
            self.solveNQueensPartialBoard(0, j, copy.deepcopy(empty_board))
        return self.solutions

    def solveNQueensPartialBoard(self, i, j, board):
        board[i][j] = OCCUPIED
        if i == self.n - 1:
            # last row
            # print_matrix(board)
            # print('-' * 10)
            self.solutions.append(self.solutionify(board))
            return

        # set new attackable cells
        self.setAttackableCells(i, j, board)

        # search for each row beneath it
        solutions = []
        for next_j in range(self.n):
            if board[i+1][next_j] == FREE:
                self.solveNQueensPartialBoard(i+1, next_j, copy.deepcopy(board))


    def setAttackableCells(self, i, j, board):
        # because we're searching row by row, there's no need to set the row as ATTACKABLE.
        # this column
        for col in range(i+1, self.n):
            board[col][j] = ATTACKABLE

        # this diagonal
        r, c = i+1, j+1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c += 1

        # this anti diagonal
        r, c = i+1, j-1
        while self.inRange(r, c):
            board[r][c] = ATTACKABLE
            r += 1
            c -= 1

    def inRange(self, r, c):
        return (0 <= r < self.n) and (0 <= c < self.n)

    def solutionify(self, board):
        out = []
        for r in board:
            s = ''
            for c in r:
                s += 'Q' if c == OCCUPIED else '.'
            out.append(s)
        return out

    def test(self):
        print(self.solveNQueens(1))
        print(self.solveNQueens(2))
        print(self.solveNQueens(3))
        print(self.solveNQueens(4))
        print(self.solveNQueens(5))


class Q160:
    # Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect.
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        # p1: traverse A, C, B
        # p2: traverse B, C, A
        # when the two pointers match, that's the intersection
        # if both pointers are end, then no intersection
        p1 = headA
        p2 = headB
        while not (p1 is None and p2 is None):

            if p1 is None:
                p1 = headB
            if p2 is None:
                p2 = headA
            if p1 == p2:
                return p1

            p1 = p1.next
            p2 = p2.next
        return None

    def test(self):
        l1_head = ll.makeLinkedList([1, 9, 1, 2, 4])
        l2_head = ListNode(val=3, next=l1_head.next.next.next)
        print_assert(self.getIntersectionNode(l1_head, l2_head), l1_head.next.next.next)

        l1_head = ll.makeLinkedList([2, 6, 4])
        l2_head = ll.makeLinkedList([1, 5])
        print_assert(self.getIntersectionNode(l1_head, l2_head), None)

class Q88:
    # Merge nums1 and nums2 into a single array sorted in non-decreasing order.
    # The final sorted array should not be returned by the function, but instead be stored inside the array nums1.
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # using two pointers from the back, gradually move numbers to the back of nums1
        i, j = m-1, n-1
        for k in reversed(range(m+n)):
            if i < 0:
                nums1[k] = nums2[j]
                j -= 1
                continue
            if j < 0:
                nums1[k] = nums1[i]
                i -= 1
                continue

            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1

    def test(self):
        nums1 = [1, 2, 3, 0, 0, 0]
        self.merge(nums1, 3, [2, 5, 6], 3)
        print_assert(nums1, [1, 2, 2, 3, 5, 6])

        nums1 = [0]
        self.merge(nums1, 0, [1], 1)
        print_assert(nums1, [1])

        nums1 = [1]
        self.merge(nums1, 1, [], 0)
        print_assert(nums1, [1])

        nums1 = [2, 0]
        self.merge(nums1, 1, [1], 1)
        print_assert(nums1, [1, 2])


class LRUCache:

    def __init__(self, capacity: int):
        self.keys = deque([])  # right is recent, left is old
        self.map = {}
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.map:
            # bump
            self.keys.remove(key)
            self.keys.append(key)
            return self.map[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            # perform an update. no one is evicted.
            # current key is bumped.
            self.map[key] = value # update
            self.keys.remove(key)
            self.keys.append(key)  # bump
        else:
            # add a new key. oldest is evicted.
            self.map[key] = value
            self.keys.append(key)
            if len(self.keys) > self.capacity:
                del self.map[self.keys.popleft()]


class Q146:
    def test(self):
        lru = LRUCache(2)
        lru.put(1, 1)
        lru.put(2, 2)
        print_assert(lru.get(1), 1)
        lru.put(3, 3)
        print_assert(lru.get(2), -1)
        lru.put(4, 4)
        print_assert(lru.get(1), -1)
        print_assert(lru.get(3), 3)
        print_assert(lru.get(4), 4)


class Q1332:
    # You are given a string s consisting only of letters 'a' and 'b'. In a single step you can remove one palindromic
    # subsequence from s.
    # Return the minimum number of steps to make the given string empty.
    def removePalindromeSub(self, s: str) -> int:
        # this is a fucking brainteaser lmao
        # if s is a palindrome, then answer is 1
        # else answer is 2, because you can remove all a's and then all b's
        return 1 if s[::-1] == s else 2

    def test(self):
        print_assert(self.removePalindromeSub('ababa'), 1)
        print_assert(self.removePalindromeSub('abb'), 2)
        print_assert(self.removePalindromeSub('baabb'), 2)

class Q3:
    # Given a string s, find the length of the longest substring without repeating characters.
    def lengthOfLongestSubstring(self, s: str) -> int:
        cur_letters = set([])
        max_len = 0
        begin = 0
        for end in range(len(s)):
            if s[end] in cur_letters:
                # advance `begin` until you reach the repeated letter
                while s[begin] != s[end]:
                    cur_letters.remove(s[begin])
                    begin += 1
                begin += 1  # start fresh
            else:
                # add this letter to the current substring
                cur_letters.add(s[end])
                max_len = max(max_len, end-begin+1)
        return max_len

    def test(self):
        print_assert(self.lengthOfLongestSubstring('abcabcbb'), 3)
        print_assert(self.lengthOfLongestSubstring('bbbbb'), 1)
        print_assert(self.lengthOfLongestSubstring('pwwkew'), 3)
        print_assert(self.lengthOfLongestSubstring(''), 0)
        print_assert(self.lengthOfLongestSubstring('abcde'), 5)


class Q1151:
    # Given a binary array data, return the minimum number of swaps required to group all 1’s present in the array
    # together in any place in the array.
    def minSwaps(self, data: List[int]) -> int:
        # first count how many 1's there are in total
        num_1s = data.count(1)

        # imagine you have a sliding windows of `num_1s` many ones.
        # you do a bitwise XOR to get the number of swaps required
        # if this window is the place where the gourp of all 1s are
        # an easier way to implement this is to only count the updates, instead of counting the whole window
        window_xor = data[:num_1s].count(0)
        min_swaps = window_xor
        for j in range(num_1s, len(data)):
            window_xor += (data[j]==0) - (data[j-num_1s]==0)
            min_swaps = min(min_swaps, window_xor)
        return min_swaps

    def test(self):
        print_assert(self.minSwaps([1, 0, 1, 0, 1]), 1)
        print_assert(self.minSwaps([0, 0, 0, 1, 0]), 0)
        print_assert(self.minSwaps([1,0,1,0,1,0,0,1,1,0,1]), 3)

class Q1197:
    # In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square [0, 0].
    # Return the minimum number of steps needed to move the knight to the square [x, y]
    def minKnightMoves(self, x: int, y: int) -> int:
        # gradually fan out from the src
        q = deque([(x, y, 0)])

        while len(q) > 0:
            cur_x, cur_y, cur_steps = q.popleft()
            if cur_x == cur_y == 0:
                return cur_steps
            for d_x, d_y in self.get_options(cur_x, cur_y):
                next_x = cur_x + d_x
                next_y = cur_y + d_y
                q.append((next_x, next_y, cur_steps + 1))

    def get_options(self, x, y):
        if abs(x) + abs(y) > 7:
            # just get closer
            if x >= 0 and y >= 0:  # go SW
                return [(-2, -1) if abs(x) > abs(y) else (-1, -2)]
            if x >= 0 and y < 0:  # go NW
                return [(-2,  1) if abs(x) > abs(y) else (-1,  2)]
            if x < 0 and y >= 0:  # go SE
                return [( 2, -1) if abs(x) > abs(y) else ( 1, -2)]
            if x < 0 and y < 0:  # go NE
                return [( 2,  1) if abs(x) > abs(y) else ( 1,  2)]
        else:
            return [( 1,  2), (-1,  2),
                    ( 1, -2), (-1, -2),
                    (-2,  1), (-2, -1),
                    ( 2,  1), ( 2, -1)]

    def test(self):
        print_assert(self.minKnightMoves(2, 1), 1)
        print_assert(self.minKnightMoves(5, 5), 4)
        print_assert(self.minKnightMoves(2, -2), 4)
        print_assert(self.minKnightMoves(-5, -4), 3)
        print_assert(self.minKnightMoves(3, 4), 3)
        print_assert(self.minKnightMoves(0, 0), 0)
        print_assert(self.minKnightMoves(2, 112), 56)


class Q1695:
    # You are given an array of positive integers nums and want to erase a subarray containing unique elements.
    # The score you get by erasing the subarray is equal to the sum of its elements.
    # Return the maximum score you can get by erasing exactly one subarray.
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        # very similar to Q3
        max_sum = 0
        cur_sum = 0
        cur_subarray = set([])
        begin = 0
        for end in range(len(nums)):
            if nums[end] in cur_subarray:
                # move begin pointer until the subarray is unique again
                while nums[begin] != nums[end]:
                    cur_sum -= nums[begin]
                    cur_subarray.remove(nums[begin])
                    begin += 1
                begin += 1
            else:
                cur_subarray.add(nums[end])
                cur_sum += nums[end]
                max_sum = max(max_sum, cur_sum)
        return max_sum

    def test(self):
        print_assert(self.maximumUniqueSubarray([4, 2, 4, 5, 6]), 17)
        print_assert(self.maximumUniqueSubarray([5, 2, 1, 2, 5, 2, 1, 2, 5]), 8)
        print_assert(self.maximumUniqueSubarray([5]), 5)
        print_assert(self.maximumUniqueSubarray([5, 2]), 7)
        print_assert(self.maximumUniqueSubarray([5, 5]), 5)


class Q1658:
    # Return the minimum number of left or right edge elements needed to sum to x. return -1 if impossible
    def minOperations(self, nums: List[int], x: int) -> int:
        subsum_target = sum(nums) - x
        if subsum_target == 0:
            return len(nums)
        # equivalent problem: find the longest subarray that sums to subsum_target
        wanted = {subsum_target: -1}
        cum_sum = 0
        longest_subarray = 0
        for idx, num in enumerate(nums):
            cum_sum += num
            if cum_sum in wanted:
                longest_subarray = max(longest_subarray, idx - wanted[cum_sum])
            wanted[cum_sum + subsum_target] = idx
        return -1 if longest_subarray == 0 else len(nums) - longest_subarray

    def test(self):
        print_assert(self.minOperations([1, 1, 4, 2, 3], 5), 2)
        print_assert(self.minOperations([5, 6, 7, 8, 9], 4), -1)
        print_assert(self.minOperations([3, 2, 20, 1, 1, 3], 10), 5)
        print_assert(self.minOperations([1, 4, 7, 2, 3, 2, 3, 2], 13), 6)
        print_assert(self.minOperations([1, 4, 7, 2, 3, 2, 3, 1], 12), 3)
        print_assert(self.minOperations([1], 1), 1)
        print_assert(self.minOperations([1, 2], 1), 1)
        print_assert(self.minOperations([2, 2], 1), -1)
        print_assert(self.minOperations([4, 3, 2], 4), 1)
        print_assert(self.minOperations([4, 3, 2], 9), 3)


## BEGIN TRIP
class TrieNode:
    def __init__(self, char: str, word_idx: int=-1, children: Optional[Dict[str, 'TrieNode']]=None):
        self.char = char
        self.children: Dict[str, 'TrieNode'] = children or {}
        self.word_idx = word_idx  # if word_idx is not 1, then it represents the end of a word here
        # this may be overwritten, but this question states
        # "If there is more than one valid index, return the largest of them."

    def __repr__(self):
        return f"TrieNode(char={self.char}, n_children={len(self.children)}, word_idx={self.word_idx})"


class Trie:
    def __init__(self):
        self.root = TrieNode("#")

    def add(self, word: str, idx: int=-1):
        cur_node = self.root
        for char in word:
            if char in cur_node.children:
                cur_node = cur_node.children[char]
                cur_node.word_idx = idx
            else:
                new_node = TrieNode(char, word_idx = idx)
                cur_node.children[char] = new_node
                cur_node = new_node

        # at this stage, the last char is recorded

    def find(self, prefix):
        cur_node = self.root
        for char in prefix:
            if char in cur_node.children:
                cur_node = cur_node.children[char]
            else:
                return -1
        return cur_node.word_idx


class WordFilter:
    def __init__(self, words: List[str]):
        # build a prefix-suffix tree here
        self.trie = Trie()
        for i, word in enumerate(words):
            # 'test' -> "_test", "t_test", "st_test", "est_test", "test_test"
            for j in range(len(word)+1):
                new_word = word[j:]+'_'+word
                self.trie.add(new_word, i)

    def f(self, prefix: str, suffix: str) -> int:
        new_prefix = suffix + '_' + prefix
        return self.trie.find(new_prefix)


class Q745:
    # 745. Prefix and Suffix Search
    # Design a special dictionary with some words that searchs the words in it by a prefix and a suffix.
    def test(self):
        wf = WordFilter(['apple'])
        print_assert(wf.f('a', 'e'), 0)
        wf = WordFilter(['apple', 'ape', 'adolf', 'hello'])
        print_assert(wf.f('a', 'e'), 1)


class Q1689:
    # 1689. Partitioning Into Minimum Number Of Deci-Binary Numbers
    def minPartitions(self, n: str) -> int:
        return int(max(n))

    def test(self):
        print_assert(self.minPartitions('32'), 3)
        print_assert(self.minPartitions('82734'), 8)
        print_assert(self.minPartitions('27346209830709182346'), 9)

if __name__ == '__main__':
    Q1689().test()
