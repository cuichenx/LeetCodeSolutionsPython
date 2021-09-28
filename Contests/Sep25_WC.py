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
    def maximumDifference(self, nums: List[int]) -> int:
        max_diff = -1
        n = len(nums)
        for i in range(n-1):
            for j in range(i, n):
                if nums[i] < nums[j]:
                    max_diff = max(max_diff, nums[j]-nums[i])
        return max_diff

    def test1(self):
        print_assert(self.maximumDifference([7, 1, 5, 4]), 4)
        print_assert(self.maximumDifference([9, 4, 3, 2]), -1)
        print_assert(self.maximumDifference([1, 5, 2, 10]), 9)

    def gridGame(self, grid: List[List[int]]) -> int:
        row0_lsums, row1_rsums = [grid[0][0]], [grid[1][-1]]
        n = len(grid[0])
        for i in range(1, n):
            row0_lsums.append(row0_lsums[-1] + grid[0][i])
            row1_rsums.append(row1_rsums[-1] + grid[1][n-1-i])
        robot1_minimizer = row0_lsums[-1]
        for i in range(n):
            robot2_would_get = max(row0_lsums[-1] - row0_lsums[i], row1_rsums[-1]-row1_rsums[-i-1])
            if robot2_would_get < robot1_minimizer:
                robot1_minimizer = robot2_would_get
        return robot1_minimizer

    def test2(self):
        print_assert(self.gridGame([[2,5,4],[1,5,1]]), 4)
        print_assert(self.gridGame([[3,3,1],[8,5,2]]), 4)
        print_assert(self.gridGame([[1,3,1,15],[1,3,3,1]]), 7)
        print_assert(self.gridGame([[1],[1]]), 0)

    def placeWordInCrossword(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        def left(i, j):
            return board[i][j-1] if j > 0 else '#'
        def right(i, j):
            return board[i][j+1] if j < n-1 else '#'
        def up(i, j):
            return board[i-1][j] if i > 0 else '#'
        def down(i, j):
            return board[i+1][j] if i < m-1 else '#'

        for i in range(m):
            for j in range(n):
                # test if word can start here horizontally right
                if left(i, j) == '#':
                    for letter_idx in range(len(word)):
                        word_pos = j + letter_idx
                        if word_pos >= n or (board[i][word_pos] != word[letter_idx] and board[i][word_pos] != ' '):
                            break
                    else:  # no break
                        if right(i, j+len(word)-1) == '#':
                            return True
                # test if word can start here horizontally left:
                if right(i, j) == '#':
                    for letter_idx in range(len(word)):
                        word_pos = j - letter_idx
                        if word_pos < 0 or (board[i][word_pos] != word[letter_idx] and board[i][word_pos] != ' '):
                            break
                    else:  # no break
                        if left(i, j-len(word)+1) == '#':
                            return True
                # test if word can start here vertically down
                if up(i, j) == '#':
                    for letter_idx in range(len(word)):
                        word_pos = i + letter_idx
                        if word_pos >= m or (board[word_pos][j] != word[letter_idx] and board[word_pos][j] != ' '):
                            break
                    else: # no break
                        if down(i + len(word)-1, j) == '#':
                            return True
                # test if word can start here vertically up
                if down(i, j) == '#':
                    for letter_idx in range(len(word)):
                        word_pos = i - letter_idx
                        if word_pos < 0 or (board[word_pos][j] != word[letter_idx] and board[word_pos][j] != ' '):
                            break
                    else: # no break
                        if up(i - len(word)+1, j) == '#':
                            return True
        return False

    def test3(self):
        print_assert(self.placeWordInCrossword([["#", " ", "#"], [" ", " ", "#"], ["#", "c", " "]], 'abc'), True)
        print_assert(self.placeWordInCrossword([[" ", "#", "a"], [" ", "#", "c"], [" ", "#", "a"]], 'ac'), False)
        print_assert(self.placeWordInCrossword([["#", " ", "#"], [" ", " ", "#"], ["#", " ", "c"]], 'ca'), True)
        print_assert(self.placeWordInCrossword([["#", " ", "#"], ["x", " ", "#"], ["#", " ", "c"]], 'ca'), True)
        print_assert(self.placeWordInCrossword([["#", " ", "#"], ["x", " ", " "], ["#", " ", "c"]], 'ca'), True)
        print_assert(self.placeWordInCrossword([["#", " ", "#"], ["#", " ", " "], ["#", " ", "c"]], 'ca'), True)
        print_assert(self.placeWordInCrossword([["#", " ", "#"], ["#", " ", "#"], ["#", " ", "c"]], 'ca'), True)

    def scoreOfStudents(self, s: str, answers: List[int]) -> int:
        correct_answer = eval(s)
        possible_answers = self.allPossibleAnswers(s)
        tot_score = 0
        for ans in answers:
            if ans == correct_answer:
                tot_score += 5
            elif ans in possible_answers:
                tot_score += 2
        return tot_score

    def allPossibleAnswers(self, s: str):
        answers = set([])
        def enumerate(tokens: List):
            if len(tokens) == 1:
                answers.add(tokens[0])
            else:
                for op_idx in range(1, len(tokens), 2):
                    enumerate(tokens[:op_idx-1] + [str(eval(''.join(tokens[op_idx-1:op_idx+2])))] + tokens[op_idx+2:])

        enumerate(list(s))
        return answers

    def test4(self):
        # print_assert(self.allPossibleAnswers('3+5*2'), {16, 13})
        # print_assert(self.allPossibleAnswers('7+3*1*2'), {20, 13})
        # print_assert(self.allPossibleAnswers('6+0*1'), {6})
        print_assert(self.allPossibleAnswers("1+9*2+8*3+7*4+6*5+5*6+4"), {})
        # print_assert(self.scoreOfStudents('7+3*1*2', [20, 13, 42]), 7)
        # print_assert(self.scoreOfStudents('3+5*2', [13,0,10,13,13,16,16]), 19)
        # print_assert(self.scoreOfStudents('6+0*1', [12,9,6,4,8,6]), 10)

if __name__ == '__main__':
    Solution().test4()


