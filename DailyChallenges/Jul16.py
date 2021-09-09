from typing import List
from bisect import bisect_left, bisect_right
from helpers import print_assert

class Solution:
    # 18 4Sum
    # Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
    # 0 <= a, b, c, d < n
    # a, b, c, and d are distinct.
    # nums[a] + nums[b] + nums[c] + nums[d] == target
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ret = set([])
        for i in range(n-3):
            for j in range(i+1, n-2):
                l = n-1  # check the fourth number from the right (biggest)
                for k in range(j+1, n-1):
                    # move l to the left while sum is larger than target
                    while nums[i] + nums[j] + nums[k] + nums[l] > target and l > k+1:
                        l -= 1
                    if nums[i] + nums[j] + nums[k] + nums[l] == target:  # bingo
                        ret.add((nums[i], nums[j], nums[k], nums[l]))
                    elif l == k+1:
                        # l and k are adjacent now: move j forward, reset k to j+1 and l to n-1
                        break
                    # else:
                    #     # too small, move k forward
                    #     continue

        return list(ret)
    # Time: O(n^3) because k and l together runs a maximum of n times
    # Space: O(logn) if quicksort, or O(n) merge sort

    # 468. Validate IP Address
    def validIPAddress(self, IP: str) -> str:
        split_dot = IP.split('.')
        if len(split_dot) == 4:  # potentially IPv4
            try:
                return 'IPv4' if all(0 <= int(addr) < 256 and (addr=='0' or not addr.startswith('0'))
                                     for addr in split_dot) else "Neither"
            except ValueError:
                return 'Neither'

        for hex in 'abcdef':
            IP = IP.lower().replace(hex, '0')
        split_colon = IP.split(':')
        if len(split_colon) == 8:  # potentially IPv6
            try:
                return 'IPv6' if all(0 <= int(addr) <= 9999 and len(addr) <= 4 for addr in split_colon) else "Neither"
            except ValueError:
                return "Neither"
        return 'Neither'


if __name__ == '__main__':
    sol = Solution()
    # print_assert(sol.fourSum([1, 0, -1, 0, -2, 2], 0),
    #              [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]])
    # print_assert(sol.fourSum([2, 2, 2, 2, 2], 8),
    #              [[2, 2, 2, 2]])
    # print_assert(len(sol.fourSum([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 0)), 24)
    #              # [[-5, -4, 4, 5],
    #              #  [-5, -3, 3, 5],
    #              #  [-5, -2, 2, 5],
    #              #  [-5, -2, 3, 4],
    #              #  [-5, -1, 1, 5],
    #              #  [-5, -1, 2, 4],
    #              #  [-5, 0, 1, 4],
    #              #  [-5, 0, 2, 3],
    #              #  [-4, -3, 2, 5],
    #              #  [-4, -3, 3, 4],
    #              #  [-4, -2, 1, 5],
    #              #  [-4, -2, 2, 4],
    #              #  [-4, -1, 0, 5],
    #              #  [-4, -1, 1, 4],
    #              #  [-4, -1, 2, 3],
    #              #  [-4, 0, 1, 3],
    #              #  [-3, -2, 0, 5],
    #              #  [-3, -2, 1, 4],
    #              #  [-3, -2, 2, 3],
    #              #  [-3, -1, 0, 4],
    #              #  [-3, -1, 1, 3],
    #              #  [-3, 0, 1, 2],
    #              #  [-2, -1, 0, 3],
    #              #  [-2, -1, 1, 2]])
    print_assert(sol.validIPAddress("172.16.254.1"), 'IPv4')
    print_assert(sol.validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334"), 'IPv6')
    print_assert(sol.validIPAddress("256.256.256.256"), 'Neither')
    print_assert(sol.validIPAddress("255.255.255.025"), 'Neither')
    print_assert(sol.validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334:"), 'Neither')
    print_assert(sol.validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:733g"), 'Neither')
    print_assert(sol.validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:73399"), 'Neither')
    print_assert(sol.validIPAddress("1e1.4.5.6"), 'Neither')
