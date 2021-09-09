from typing import List

from helpers import print_assert
from collections import deque
# 384 Shuffle an array
# Given an integer array nums, design an algorithm to randomly shuffle the array.
# All permutations of the array should be equally likely as a result of the shuffling.
## this was the most useless question ever. thank you next
import random
class Solution:
    def __init__(self, nums: List[int]):

        self.array = nums
        self.orig_array = nums

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        self.array = self.orig_array.copy()
        return self.array

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        self.array = self.orig_array.copy()
        random.shuffle(self.array)
        return self.array

# 55. Jump Game
# Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
# Each element in the array represents your maximum jump length at that position.
# Determine if you are able to reach the last index.
class JumpGame:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        target = 1
        for i in reversed(range(n-1)):
            if nums[i] >= target:
                target = 1  # win condition becomes reaching nums[i]
            else:
                target += 1
        return target == 1  # if win condition eventually becomes reaching nums[0], return true
    # Time: O(N), Space: O(1)

# 45. Jump Game II
# Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
# Each element in the array represents your maximum jump length at that position.
# Your goal is to reach the last index in the minimum number of jumps.
# You can assume that you can always reach the last index.
    def jump(self, nums: List[int]) -> int:
        # BST from first integer, until last elem is reached
        n = len(nums)
        q = deque([(0, 0)])  # (elem idx, depth)
        visited = [False] * n
        while True:
            head, depth = q.popleft()
            if head == n-1:
                return depth
            for i in reversed(range(head+1, min(n, head+1+nums[head]))):
                # if i == n-1:
                #     return depth+1
                if not visited[i]:
                    visited[i] = True
                    q.append((i, depth+1))
    # Time: O(N), Space: O(N)

    def jump_fast(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 1:
            return 0
        midx = 0
        r = range(0, 1)
        count = 0
        ln = len(nums)
        while midx < ln - 1:
            for p in reversed(r):
                midx = max(midx, p + nums[p])
                if midx >= ln - 1:
                    return count + 1
                r = range(r[0] + 1, midx + 1)
            count += 1
        return count


if __name__ == '__main__':
    # nums = [1, 2, 3, 4, 5]
    # obj = Solution(nums)
    # param_1 = obj.reset()
    # param_2 = obj.shuffle()
    jg = JumpGame()
    # print_assert(jg.canJump([2, 3, 1, 1, 4]), True)
    # print_assert(jg.canJump([3, 2, 1, 0, 4]), False)
    # print_assert(jg.canJump([2, 0, 2, 4, 0, 0, 0, 8]), True)
    # print_assert(jg.canJump([1, 0, 0, 0, 0, 0, 0, 8]), False)
    print_assert(jg.jump([2, 3, 1, 1, 4]), 2)
    print_assert(jg.jump([2, 3, 0, 1, 4]), 2)
    print_assert(jg.jump([2, 0, 2, 4, 0, 0, 0, 8]), 3)
    print_assert(jg.jump([2]), 0)
    print_assert(jg.jump([2, 0]), 1)
    print_assert(jg.jump([2, 0, 2]), 1)
    print_assert(jg.jump([2, 1, 1, 2]), 2)
    import time
    t0 = time.time()
    print_assert(jg.jump([1,2,1,1,1,4,4,1,5,2,3,4,1,4,2,5,2,6,4,4,2,2,5,6,2,3,4,5,4,4,2,3,1,4,1,6,2,3,5,3,6,6,1,2,5,3,3,4,6,1,1,5,3,3,4,5,1,4,2,6,6,4,1,4,1,2,1,4,4,2,1,2,2,5,6,5,4,4,3,6,5,2,5,6,1,4,3,4,3,3,1,2,6,5,3,6,1,2,6,4,2,3,3,4,6,3,5,3,2,3,3,1,3,2,4,1,3,5,1,1,5,2,4,2,2,5,3,4,2,1,3,3,1,2,4,5,4,6,2,5,6,4,6,5,2,2,1,4,6,4,2,4,1,6,3,3,6,1,4,5,4,5,1,2,3,6,1,4,3,2,5,1,5,2,5,1,2,3,3,6,6,3,5,2,6,1,6,4,3,4,1,2,5,1,5,6,5,3,1,5,6,3,6,3,5,6,2,2,6,3,4,1,4,1,1,3,4,1,5,6,5,4,2,5,3,6,4,1,2,3,5,6,5,2,3,6,1,3,4,6,3,2,5,5,1,6,6,6,2,3,5,5,4,5,2,1,6,6,2,5,1,3,2,5,1,2,3,4,1,1,5,1,4,1,2,2,6,1,4,3,2,1,6,5,1,6,2,3,5,3,6,6,5,2,1,4,4,5,3,5,5,1,3,2,6,1,6,6,4,6,5,3,3,1,6,2,6,4,2,4,1,2,2,2,2,1,5,4,3,6,3,2,5,5,4,6,4,1,5,2,4,6,2,4,5,5,3,4,6,6,1,6,6,5,3,1,4,6,5,3,5,3,5,2,3,4,6,2,5,6,6,2,5,6,1,1,5,4,5,6,6,5,5,3,3,4,4,5,2,6,5,1,3,2,3,1,3,1,2,3,5,2,5,3,2,2,3,4,4,2,6,5,1,3,4,6,1,6,4,4,2,4,5,2,5,6,6,1,3,1,1,4,6,5,6,4,1,3,1,1,6,2,6,4,5,5,3,5,3,6,6,2,1,3,2,5,5,3,5,3,3,5,3,2,1,2,2,6,1,6,4,2,2,2,6,2,4,2,5,5,2,3,1,1,5,6,6,3,4,6,2,1,2,1,4,2,5,6,5,5,3,2,1,5,1,3,2,2,5,1,6,1,6,5,6,2,6,3,6,5,1,4,6,3,3,6,6,4,1,4,6,3,4,1,4,2,5,5,5,4,2,5,6,6,3,1,5,4,2,3,6,1,6,4,1,5,5,6,4,5,4,4,6,5,2,5,1,4,3,2,6,1,5,2,6,2,6,1,2,3,5,5,4,4,5,4,2,1,4,1,4,6,1,1,2,6,2,3,6,4,4,5,6,6,4,1,6,3,2,4,1,4,5,5,2,6,6,4,2,5,4,6,6,5,2,4,1,1,4,1,1,4,6,1,5,2,4,6,5,1,6,6,6,2,1,6,1,5,5,4,5,2,3,2,2,2,6,4,6,2,4,6,4,5,1,3,2,4,2,6,6,4,3,3,1,1,4,4,5,5,4,1,6,5,1,3,3,6,5,5,3,6,3,5,2,4,3,4,6,5,2,6,6,1,2,3,4,6,1,5,6,4,6,6,1,1,2,4,6,4,1,1,6,6,2,1,1,2,3,6,5,3,1,6,1,3,6,2,4,5,3,2,5,3,5,5,2,1,3,4,4,6,2,4,3,3,1,5,3,3,1,2,5,2,5,2,2,4,2,2,4,6,3,1,4,2,3,4,2,2,6,3,2,6,3,3,5,5,5,2,3,1,6,5,4,5,2,6,5,2,1,2,2,2,2,2,3,2,6,3,1,5,6,1,4,6,5,3,3,5,5,6,5,1,4,3,5,5,3,4,6,4,6,3,2,1,1,6,2,2,5,5,3,1,3,5,6,3,6,2,5,6,2,1,4,4,2,2,6,2,1,5,6,1,1,3,3,5,5,3,2,5,2,1,3,2,4,3,5,2,5,5,4,1,1,3,4,3,1,3,5,5,4,5,5,1,3,5,4,6,5,4,2,1,2,6,6,4,4,5,6,6,6,3,4,3,5,2,5,6,5,2,1,4,5,3,1,6,4,1,5,4,5,2,5,1,4,2,6,3,3,5,1,3,4,3,3,6,6,5,5,5,4,5,3,6,6,6,4,2,4,4,1,2,2,2,3,2,2,5,6,5,6,3,3,1,1,4,1,6,6,5,3,2,6,5,2,1,6,1,4,6,4,1,2,1,2,5,1,1,6,3,2,5,4,5,2,6,5,6,2,2,1,5,5,1,6,2,1,3,4,5,4,3,1,5,6,5,4,1,2,3,4,2,2,6,2,4,3,2,5,3,2,2,5,6,3,3,2,1,4,5,2,3,2,5,3,1,3,6,3,6,4,2,5,3,6,1,6,5,2,1,5,2,1,1,4,3,3,1,1,2,2,1,1,4,1,6,5,5,6,4,6,6,2,2,2,6,1,1,1,1,5,2,2,1,6,5,6,1,3,1,6,4,1,2,1,5,1,1,3,6,4,5,4,2,3,4,1,5,2,2,1,6,2,3,2,3,3,1,1,4,5,5,3,5,3,6,4,5,4,4,4,2,2,1,4,6,0,0,0,0,0]), 9999)
    print(time.time()-t0, 'seconds')
