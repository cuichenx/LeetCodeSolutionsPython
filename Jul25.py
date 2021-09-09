from typing import List

from helpers import print_assert
import math

class Solution:
    # 600. Non-negative Integers without Consecutive Ones
    # Given a positive integer n, return the number of the integers in the range [0, n]
    # whose binary representations do not contain consecutive ones.
    def __init__(self):
        self.Nk = None

    def findIntegers(self, n: int) -> int:
        # if n < 0:
        #     return 0
        if n <= 2:
            return n+1
        k = int(math.log2(n))
        # let 0 < k <= K be the number of digits in the binary representation
        # let N(k) be the number of integers without consecutive ones
        # then: N(k) = N(1) + N(2) + ... + N(k-2)
        # N(1) = 2, N(2) = 1
        # after finding N(k), we know the number of integers in the interval [0, 2^k - 1]
        # to find the number of integers between [2^k, n], subtract by 2^k and calculate with the same way

        # step 1: establish fibbonacci-like sequence
        # if self.Nk is None:
        Nk = [2, 1, 2]
        for _ in range(k):
            # Nk.append(Nk[0] + Nk[1] + ... + Nk[i-2]
            Nk.append(Nk[-1] + Nk[-2])

        # step 2: find number of remaining integers between 10xxxx and n
        # consider two cases: numbers starting with 11xxxx and numbers less than 101111
        # in the first case, the subproblem is the number of integers in 2**(k-1)-1
        # in the second case, the subproblem is the number of integers in n-2**k
        # in both cases, it's the binary number with two leftmost digits cut off
        return Nk[k+1] + self.findIntegers(min(2**(k-1)-1, n-2**k))


    # 342. Power of Four
    # Given an integer n, return true if it is a power of four. Otherwise, return false.
    def isPowerOfFour(self, n: int) -> bool:
        if n == 1:
            return True

        while n > 1:
            n /= 4
            if n == 1:
                return True
            # elif n < 1:
            #     return False
            #
        # if n < 1, return False
        return False


# 304. Range Sum Query 2D - Immutable
# Given a 2D matrix matrix, handle multiple queries of the following type:
#
# Calculate the sum of the elements of matrix inside the rectangle defined by its upper left corner (row1, col1)
# and lower right corner (row2, col2).
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        v_sum = [0] * n
        cumul = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                cumul[i][j] = (cumul[i][j-1] if j > 0 else 0) + matrix[i][j] + v_sum[j]
                v_sum[j] += matrix[i][j]
        self.cumul = cumul

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.cumul[row2][col2] \
               - (self.cumul[row2][col1-1] if col1 > 0 else 0) \
               - (self.cumul[row1-1][col2] if row1 > 0 else 0) \
               + (self.cumul[row1-1][col1-1] if row1 > 0 and col1 > 0 else 0)


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)

if __name__ == '__main__':
    sol = Solution()
    # print_assert(actual=sol.findIntegers(1), expected=2)
    # print_assert(actual=sol.findIntegers(2), expected=3)
    # print_assert(actual=sol.findIntegers(3), expected=3)
    # print_assert(actual=sol.findIntegers(4), expected=4)
    # print_assert(actual=sol.findIntegers(5), expected=5)
    # print_assert(actual=sol.findIntegers(6), expected=5)
    # print_assert(actual=sol.findIntegers(7), expected=5)
    # print_assert(actual=sol.findIntegers(8), expected=6)
    # print_assert(actual=sol.findIntegers(9), expected=7)
    # print_assert(actual=sol.findIntegers(10), expected=8)
    # print_assert(actual=sol.findIntegers(11), expected=8)
    # print_assert(actual=sol.findIntegers(12), expected=8)
    # print_assert(actual=sol.findIntegers(13), expected=8)
    # print_assert(actual=sol.isPowerOfFour(16), expected=True)
    # print_assert(actual=sol.isPowerOfFour(8), expected=False)
    # print_assert(actual=sol.isPowerOfFour(4), expected=True)
    # print_assert(actual=sol.isPowerOfFour(2), expected=False)
    # print_assert(actual=sol.isPowerOfFour(1), expected=True)
    # print_assert(actual=sol.isPowerOfFour(-2), expected=False)
    # print_assert(actual=sol.isPowerOfFour(-4), expected=False)
    nm = NumMatrix([[3, 0, 1, 4, 2],
                    [5, 6, 3, 2, 1],
                    [1, 2, 0, 1, 5],
                    [4, 1, 0, 1, 7],
                    [1, 0, 3, 0, 5]])
    print_assert(actual=nm.sumRegion(2, 1, 4, 3), expected=8)
    print_assert(actual=nm.sumRegion(1, 1, 2, 2), expected=11)
    print_assert(actual=nm.sumRegion(1, 2, 2, 4), expected=12)
    nm2 = NumMatrix([[-1]])
    print_assert(actual=nm2.sumRegion(0, 0, 0, 0), expected=-1)
