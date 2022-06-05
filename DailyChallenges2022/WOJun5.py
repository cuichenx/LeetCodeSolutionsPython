import sys
sys.path.append("..")
from helpers import print_assert, print_matrix
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

if __name__ == '__main__':
    Q52().test()
