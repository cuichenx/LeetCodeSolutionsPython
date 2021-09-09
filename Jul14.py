# Definition for a binary tree node.
from typing import Tuple, List, Union
from helpers import TreeNode, Tree as tr


class Solution:
    # 791
    def customSortString(self, order: str, str: str) -> str:
        order_map = [0] * 26
        for i, c in enumerate(order):
            order_map[ord(c) - ord('a')] = i

        return ''.join(sorted(str, key=lambda c: order_map[ord(c) - ord('a')]))

    # 3
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) <= 1:
            return len(s)
        beg = 0
        end = 0
        cur_seq_letters = [-1] * 256  # cur_seq_letters[x] = y means that the letter chr(x) is in the yth position
        cur_max = 0
        while end < len(s):
            end_ascii = ord(s[end])
            i = cur_seq_letters[end_ascii]
            if i != -1 and beg <= i < end:
                # repeated char in this sequence. record the length of this sequence
                beg = cur_seq_letters[end_ascii] + 1  # set beg to this repeated char + 1
                # end = beg
                # cur_seq_letters = [-1] * 256
            cur_max = max(cur_max, end - beg + 1)
            # else:
            # still valid
            cur_seq_letters[end_ascii] = end
            end += 1
        # cur_max = max(cur_max, end - beg)
        return cur_max

    # 1530
    def countPairs(self, root: TreeNode, distance: int) -> int:
        return self.countChildren(root, distance)[-1]

    def countPairsIO(self, tree: List, distance: int) -> int:
        return self.countPairs(tr.list2tree(tree), distance)

    def countChildren(self, root: TreeNode, distance: int) -> Tuple[Union[List[int], None], int]:
        if root is None:
            return None, 0

        lc_d2l, lc_count = self.countChildren(root.left, distance)
        rc_d2l, rc_count = self.countChildren(root.right, distance)
        if lc_d2l is None and rc_d2l is None:
            # i am a leaf node
            return [0], 0
        elif lc_d2l is None:
            # only has right child
            return [c + 1 for c in rc_d2l], lc_count + rc_count
        elif rc_d2l is None:
            # only has left child
            return [c + 1 for c in lc_d2l], lc_count + rc_count
        else:
            # has both left and right children
            # check whether any pair of children can link
            left_distances = [c + 1 for c in lc_d2l]
            right_distances = [c + 1 for c in rc_d2l]
            my_count = self.check_children_distances(left_distances, right_distances, distance)
            return left_distances + right_distances, lc_count + rc_count + my_count

    def check_children_distances(self, left_distances, right_distances, distance):
        count = 0
        for l in left_distances:
            for r in right_distances:
                if l + r <= distance:
                    count += 1
        return count





if __name__ == '__main__':
    sol = Solution()
    # print(sol.customSortString(order='cba', str='abcd'))
    # print(sol.customSortString(order='adc', str='abcd'))
    # print(sol.customSortString(order='abcdefghijklmnopqrstuvwxyz', str='alskfhjlks'))
    # print(sol.customSortString(order='abcdefg', str='aaaabbbbeeeeeefccc'))
    # print(sol.lengthOfLongestSubstring('abcabcbb'), 3)
    # print(sol.lengthOfLongestSubstring('bbbbb'), 1)
    # print(sol.lengthOfLongestSubstring('pwwkew'), 3)
    # print(sol.lengthOfLongestSubstring(''), 0)
    # print(sol.lengthOfLongestSubstring('au'), 2)
    # print(sol.lengthOfLongestSubstring('abba'), 2)
    print(sol.countPairsIO([7,1,4,6,None,5,3,None,None,None,None,None,2], 3), 1)
    print(sol.countPairsIO([100], 1), 0)
    print(sol.countPairsIO([1, 1, 1], 2), 1)
    print(sol.countPairsIO( [1,2,3,None,4], 3), 1)
    print(sol.countPairsIO([9,7,8,1,4,None,None,6,None,5,3,None,None,None,None,None,2], 4), 4)


