from typing import Optional, List

from helpers import TreeNode, Tree as tr, print_assert


class Mock1:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            root.right, root.left = root.left, root.right
            self.invertTree(root.left)
            self.invertTree(root.right)
        return root

    def test1(self):
        print_assert(tr.tree2list(self.invertTree(tr.list2tree([4, 2, 7, 1, 3, 6, 9]))), [4, 7, 2, 9, 6, 3, 1])
        print_assert(tr.tree2list(self.invertTree(tr.list2tree([2, 1, 3]))), [2, 3, 1])
        print_assert(tr.tree2list(self.invertTree(tr.list2tree([]))), [])

    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        # if root is too big, the RST doesn't exist.
        if not root:
            return None
        if root.val > high:
            return self.trimBST(root.left, low, high)
        elif root.val < low:
            return self.trimBST(root.right, low, high)
        else:
            # root value is acceptable
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
            return root

    def combinations(self, arrays):
        if len(arrays) == 0:
            return []
        i = 0
        while i < len(arrays) and len(arrays[i]) == 0:
            i += 1
        if i == len(arrays):
            return []

        ret = [[elem] for elem in arrays[i]]

        for subarray in arrays[i + 1:]:
            if not subarray:
                continue
            ret_copy = ret.copy()
            ret = []
            for elem in subarray:
                ret += [r + [elem] for r in ret_copy]

        return ret

    def combinations_recur(self, arrays):
        ret = []

        def dfs(cur_comb, subarray_idx):
            if subarray_idx == len(arrays):
                # full
                ret.append(cur_comb)
                return
            if not arrays[subarray_idx]:
                # empty list. skip this
                dfs(cur_comb, subarray_idx+1)
            for elem in arrays[subarray_idx]:
                dfs(cur_comb+[elem], subarray_idx+1)
        dfs([], 0)
        return ret


    def test2(self):
        print(self.combinations_recur([[1, 2], [3, 4]]))
        print(self.combinations_recur([[1, 2], [], [3, 4]]))
        print(self.combinations_recur([[], [], [], [1, 2], [], [3, 4]]))
        print(self.combinations_recur([[], [5, 6, 7], [], [1, 2], [], [3, 4]]))

    # 17. Letter Combinations of a Phone Number  # backtracking
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        ret = []
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

        def dfs(cur_str, next_digits):
            if not next_digits:
                ret.append(cur_str)
                return

            for next_letter in mapping[next_digits[0]]:
                dfs(cur_str+next_letter, next_digits[1:])

        dfs('', digits)
        return ret

    def test3(self):
        print_assert(self.letterCombinations('23'), ["ad","ae","af","bd","be","bf","cd","ce","cf"])
        print_assert(self.letterCombinations(''), [])
        print_assert(self.letterCombinations('2'), ['a', 'b', 'c'])

if __name__ == '__main__':
    Mock1().test2()