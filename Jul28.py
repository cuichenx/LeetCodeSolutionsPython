from typing import List

from helpers import print_assert


class Solution:
    # 932. Beautiful Array
    # For some fixed n, an array nums is beautiful if it is a permutation of the integers 1, 2, ..., n, such that:
    #
    # For every i < j, there is no k with i < k < j such that nums[k] * 2 = nums[i] + nums[j].
    #
    # Given n, return any beautiful array nums.  (It is guaranteed that one exists.)
    def beautifulArray(self, N):
        memo = {1: [1]}
        def f(N):
            if N not in memo:
                odds = f((N+1)//2)
                evens = f(N//2)
                memo[N] = [2*x-1 for x in odds] + [2*x for x in evens]
            return memo[N]
        return f(N)


if __name__ == '__main__':
    sol = Solution()
    for i in range(1, 21):
        print(i, sol.beautifulArray(i))