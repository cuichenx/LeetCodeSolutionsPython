import heapq
import itertools
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Tuple

from helpers import print_assert, TreeNode, Tree as tr, Node, NaryTree as ntr


class Q415:
    # Add Strings
    # Given two non-negative integers, num1 and num2 represented as string, return the sum of num1 and num2 as a string.
    #
    # You must solve the problem without using any built-in library for handling large integers (such as BigInteger).
    # You must also not convert the inputs to integers directly.
    def addStrings(self, num1: str, num2: str) -> str:
        l1, l2 = len(num1), len(num2)
        carry = 0
        res = ''
        for i in range(1, min(l1, l2) + 1):
            s = ord(num1[-i]) - ord('0') + ord(num2[-i]) - ord('0') + carry
            carry, digit = s // 10, s % 10
            res = str(digit) + res
        if l1 == l2:
            return '1' + res if carry else res

        if l1 > l2:
            longer, shorter = num1, num2
        else:
            longer, shorter = num2, num1
        for i in range(len(shorter) + 1, len(longer) + 1):
            s = ord(longer[-i]) - ord('0') + carry
            carry, digit = s // 10, s % 10
            res = str(digit) + res

        return '1' + res if carry else res

    def test(self):
        print_assert(self.addStrings("11", "123"), "134")
        print_assert(self.addStrings("456", "77"), "533")
        print_assert(self.addStrings("0", "0"), "0")
        print_assert(self.addStrings("9999999", "1"), "10000000")
        print_assert(self.addStrings("9999999", "11"), "10000010")
        print_assert(self.addStrings("1", "9999999"), "10000000")
        print_assert(self.addStrings("0", "9999999"), "9999999")


class Q276:
    # Paint Fence
    # You are painting a fence of n posts with k different colors. You must paint the posts following these rules:
    #
    # Every post must be painted exactly one color.
    # There cannot be three or more consecutive posts with the same color.
    # Given the two integers n and k, return the number of ways you can paint the fence.
    def numWays(self, n: int, k: int) -> int:
        if n <= 2:
            return k ** n
        # consider the colours of the last two fences
        same_color, diff_color = k, k * k - k
        for _ in range(3, n + 1):
            same_color, diff_color = diff_color, (k - 1) * (same_color + diff_color)
        return same_color + diff_color

    def test(self):
        print_assert(self.numWays(3, 2), 6)
        print_assert(self.numWays(1, 1), 1)
        print_assert(self.numWays(7, 2), 42)


class Q443:
    # 443. String Compression
    def compress(self, chars: List[str]) -> int:
        if len(chars) == 1:
            return 1  # no modification

        read, write = 1, 1
        cnt = 1
        prev_char = chars[0]
        while read < len(chars):
            if chars[read] != prev_char:
                if cnt > 1:
                    for cnt_digit in str(cnt):
                        chars[write] = cnt_digit  # dump the count of previous char
                        write += 1
                chars[write] = prev_char = chars[read]
                write += 1
                cnt = 1
            else:
                cnt += 1
            read += 1

        if cnt > 1:
            for cnt_digit in str(cnt):
                chars[write] = cnt_digit  # dump the count of previous char
                write += 1

        return write

    def test(self):
        inp = ["a", "a", "b", "b", "c", "c", "c"]
        print_assert(self.compress(inp), 6)
        print_assert(inp[:6], ["a", "2", "b", "2", "c", "3"])

        inp = ["a"]
        print_assert(self.compress(inp), 1)
        print_assert(inp[:1], ["a"])

        inp = ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]
        print_assert(self.compress(inp), 4)
        print_assert(inp[:4], ["a", "b", "1", "2"])

        inp = ["a", "a", "a", "b", "b", "a", "a"]
        print_assert(self.compress(inp), 6)
        print_assert(inp[:6], ["a", "3", "b", "2", "a", "2"])

        inp = ["a", "a", "a", "b", "b", "a", "a"] + ['x'] * 104
        print_assert(self.compress(inp), 10)
        print_assert(inp[:10], ["a", "3", "b", "2", "a", "2", 'x', '1', '0', '4'])

        inp = ["a"] * 1002 + ['x'] * 104
        print_assert(self.compress(inp), 9)
        print_assert(inp[:9], ["a", "1", "0", "0", "2", 'x', '1', '0', '4'])


class Q926:
    # 926. Flip String to Monotone Increasing
    # A binary string is monotone increasing if it consists of some number of 0's (possibly none),
    # followed by some number of 1's (also possibly none).
    #
    # You are given a binary string s. You can flip s[i] changing it from 0 to 1 or from 1 to 0.
    #
    # Return the minimum number of flips to make s monotone increasing.
    def minFlipsMonoIncr(self, s: str) -> int:
        left_ones = [0]
        right_zeros = [0]  # reversed order, from right to left
        for i in range(len(s)):
            left_ones.append(left_ones[-1] + int(s[i]))
            right_zeros.append(right_zeros[-1] + int(s[-1 - i] == '0'))
        return min(l + r for l, r in zip(left_ones, reversed(right_zeros)))

    # Time: O(N), Space O(N)

    def test(self):
        print_assert(self.minFlipsMonoIncr("00100"), 1)
        print_assert(self.minFlipsMonoIncr("010110"), 2)
        print_assert(self.minFlipsMonoIncr("00011000"), 2)


class Q954:
    # 954. Array of Doubled Pairs
    # Given an integer array of even length arr, return true if it is possible to reorder arr such that
    # arr[2 * i + 1] = 2 * arr[2 * i] for every 0 <= i < len(arr) / 2, or false otherwise.
    def canReorderDoubled_1(self, arr: List[int]) -> bool:
        arr.sort()
        waitlist = defaultdict(int)
        for elem in arr:
            if waitlist[elem] > 0:
                waitlist[elem] -= 1
            else:
                if elem > 0:
                    waitlist[elem * 2] += 1
                else:
                    waitlist[elem / 2] += 1

        return sum(waitlist.values()) == 0
    # Time: O(N logN), Space: O(N)

    def canReorderDoubled(self, arr: List[int]) -> bool:
        arr.sort(key=abs)  # sort as e.g. [2, -2, 4, -4, 8, -8]
        waitlist = defaultdict(int)
        for elem in arr:
            if waitlist[elem] > 0:
                waitlist[elem] -= 1
            else:
                waitlist[elem * 2] += 1

        return sum(waitlist.values()) == 0
    # Time: O(N logN), Space: O(N)

    def test(self):
        print_assert(self.canReorderDoubled([3, 1, 3, 6]), False)
        print_assert(self.canReorderDoubled([2, 1, 2, 6]), False)
        print_assert(self.canReorderDoubled([4, -2, 2, -4]), True)
        print_assert(self.canReorderDoubled([1, 2, 4, 16, 8, 4]), False)
        print_assert(self.canReorderDoubled([1, 2, 4, 16, 8, 8]), True)
        print_assert(self.canReorderDoubled([1, 2, 4, 16, 8, 2]), True)
        print_assert(self.canReorderDoubled([2, 1, 2, 1, 1, 1, 2, 2]), True)


class Q49:
    # Group Anagrams
    # Given an array of strings strs, group the anagrams together. You can return the answer in any order.
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        words = defaultdict(list)  # keys: sorted tuple of letters. values: word
        for s in strs:
            words[tuple(sorted(list(s)))].append(s)
        return list(words.values())

    def test(self):
        print_assert(self.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]),
                     [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']])
        print_assert(self.groupAnagrams([""]), [[""]])
        print_assert(self.groupAnagrams(["a"]), [["a"]])


class Q73:
    # Set Matrix Zeroes
    # Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's, and return the matrix.
    #
    # You must do it in place.
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        first_col, first_row = False, False
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    if i == 0:
                        first_row = True
                    if j == 0:
                        first_col = True
                    else:
                        matrix[i][0] = 0
                        matrix[0][j] = 0
        # do everything except the first row and first column
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        # first row and first column become 1 only if matrix[0][0] == 0
        if first_col:
            for i in range(m):
                matrix[i][0] = 0
        if first_row:
            for j in range(n):
                matrix[0][j] = 0

    def test(self):
        matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.setZeroes(matrix)
        print_assert(matrix, [[1, 0, 1], [0, 0, 0], [1, 0, 1]])

        matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
        self.setZeroes(matrix)
        print_assert(matrix, [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]])

        matrix = [[4, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
        self.setZeroes(matrix)
        print_assert(matrix, [[0, 0, 0, 0], [3, 4, 5, 0], [1, 3, 1, 0]])


class Q546:
    # Remove Boxes
    # You are given several boxes with different colors represented by different positive numbers.
    #
    # You may experience several rounds to remove boxes until there is no box left. Each time you can choose some
    # continuous boxes with the same color (i.e., composed of k boxes, k >= 1), remove them and get k * k points.
    #
    # Return the maximum points you can get.
    def removeBoxes(self, boxes: List[int]) -> int:
        if len(boxes) <= 1:
            return len(boxes)

        # find the largest color, and make sure you get that color as a whole group
        counts = defaultdict(list)
        largest_count = 0
        for i, box_color in enumerate(boxes):
            counts[box_color].append(i)
            largest_count = max(largest_count, len(counts[box_color]))
        largest_color_candidates = [color for color, indices in counts.items() if len(indices) == largest_count]

        best_score = 0
        for largest_color in largest_color_candidates:
            seps = counts[largest_color]

            score = 0
            # remove groups of boxes in between the largest color
            for i in range(len(seps)-1):
                score += self.removeBoxes(boxes[seps[i]+1 : seps[i+1]])
            # remove this largest color
            score += largest_count ** 2
            # remove the remaining boxes, surrounding the largest color
            score += self.removeBoxes(boxes[:seps[0]] + boxes[seps[-1]+1:])
            best_score = max(best_score, score)
            # if the tie is between multiple groups of maximum 2s, there's no need to calculate multiple choices
            if largest_count < 3:
                break
        return best_score

    def test(self):
        print_assert(self.removeBoxes([1, 3, 2, 2, 2, 3, 4, 3, 1]), 23)
        print_assert(self.removeBoxes([1, 1, 1]), 9)
        print_assert(self.removeBoxes([1]), 1)
        print_assert(self.removeBoxes([20, 12, 2, 11, 6, 18, 4, 6, 8, 12, 16, 18, 15, 6, 10, 8, 20, 8, 15, 16]), 30)
        print_assert(self.removeBoxes([2, 9, 7, 6, 3, 2, 7, 6, 5, 4, 6, 4, 4, 2, 3, 3]), 30)
        print_assert(self.removeBoxes(
            [1, 1, 1, 2, 4, 8, 1, 9, 1, 2, 9, 7, 6, 3, 2, 7, 6, 5, 4, 6, 4, 4, 2, 3, 3, 1, 7, 8, 6, 9, 1, 1, 8, 10, 1,
             4, 6, 7, 7, 1, 6, 10, 7, 7, 8, 6, 1, 5, 4, 3]), 180)

if __name__ == '__main__':
    Q546().test()
