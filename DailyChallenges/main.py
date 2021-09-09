from typing import List


class Solution:

    def colorTheGrid(self, m: int, n: int) -> int:
        # traverse the grid diagonally

        board = [[-1]*n for _ in range(m)]
        board[0][0] = 0  # set top left cornor to R. multiply final answer by 3

        self.memoize = {}  # dictionary that stores (n_uncolored, frontline) as keys in case it comes up again

        if n == 1:
            return (3*(self.colorPartial(board, 1, 0))) % int(1e9+7)
        return (3*(self.colorPartial(board, 0, 1))) % int(1e9+7)

    def colorPartial(self, board, i, j):
        m, n = len(board), len(board[0])

        if i >= m:  # past the bottom right corner
            return 1

        RGB = [True, True, True]
        if i > 0:
            RGB[board[i-1][j]] = False
        if j > 0:
            RGB[board[i][j-1]] = False

        c = 0
        if i == m - 1:  # bottom edge:
            next_j = n - 1
            next_i = i + j + 1 - next_j
            if next_i < 0:  # then same logic as left edge
                next_i = 0
                next_j = i + j + 1
        elif j == 0:  # left edge:
            next_i = 0
            next_j = i + j + 1
            if next_j >= n:  # then same logic as bottom edge
                next_j = n - 1
                next_i = i + j + 1 - next_j
        else:  # not at the edge
            next_i = i+1
            next_j = j-1

        for rgb in range(3):
            if RGB[rgb]:
                board[i][j] = rgb
                memo = self.memoized_board(board)
                if memo > 0:
                    c += memo
                else:
                    c += self.colorPartial(board[:], next_i, next_j)

        return c

    def memoized_board(self, board):
        n_uncolored = 0
        frontline = []
        for i in range(board):
            for j in range(board[0]):
                if board[i][j] == -1:
                    # the square above it is front line
                    if i > 0:
                        frontline.append(board[i-1][j])

if __name__ == '__main__':
    sol = Solution()
    # print(sol.colorTheGrid(1, 1), 3)
    # print(sol.colorTheGrid(1, 2), 6)
    # print(sol.colorTheGrid(5, 5), 580986)
    print(sol.colorTheGrid(2, 37), None)

