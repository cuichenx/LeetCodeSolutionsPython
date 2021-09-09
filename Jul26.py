from typing import List, Optional

from helpers import print_assert, TreeNode, Tree as tr


class Solution:
    # 108. Convert Sorted Array to Binary Search Tree
    #
    # Given an integer array nums where the elements are sorted in ascending order,
    # convert it to a height-balanced binary search tree.
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if len(nums) == 0:
            return None
        mid = len(nums)//2
        return TreeNode(val=nums[mid],
                        left=self.sortedArrayToBST(nums[:mid]),
                        right=self.sortedArrayToBST(nums[mid+1:]))


if __name__ == '__main__':
    sol = Solution()
    null = None
    print_assert(actual=tr.tree2list(sol.sortedArrayToBST([-10, -3, 0, 5, 9])),
                 expected=([0, -3, 9, -10, null, 5], [0, -10, 5, null, -3, null, 9]))
    print_assert(actual=tr.tree2list(sol.sortedArrayToBST([1, 3])),
                 expected=([1, 3], [3, 1]))
