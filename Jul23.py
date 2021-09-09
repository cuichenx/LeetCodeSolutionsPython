from typing import Optional

from helpers import print_assert, TreeNode, Tree as tr


class Solution:
    # 814. Binary Tree Pruning
    # Given the root of a binary tree, return the same tree where every subtree (of the given tree)
    # not containing a 1 has been removed.
    def pruneTree(self, root: TreeNode) -> Optional[TreeNode]:
        if self.prune(root):
            return None
        return root

    def prune(self, root: TreeNode) -> bool:
        # prunes subtrees not containing any ones, and return whether pruning can be continued one level up
        if root is None:
            return True
        l = self.prune(root.left)
        if l: root.left = None
        r = self.prune(root.right)
        if r: root.right = None
        if l and r and root.val == 0:
            # can continue pruning
            return True
        else:
            return False


if __name__ == '__main__':
    sol = Solution()
    null = None
    print_assert(actual=tr.tree2list(sol.pruneTree(tr.list2tree([1, null, 0, 0, 1]))),
                 expected=[1, null, 0, null, 1])
    print_assert(actual=tr.tree2list(sol.pruneTree(tr.list2tree([1, 0, 1, 0, 0, 0, 1]))),
                 expected=[1, null, 1, null, 1])
    print_assert(actual=tr.tree2list(sol.pruneTree(tr.list2tree([1, 1, 0, 1, 1, 0, 1, 0]))),
                 expected=[1, 1, 0, 1, 1, null, 1])
    print_assert(actual=tr.tree2list(sol.pruneTree(tr.list2tree([0, 0, 0, null, 0]))),
                 expected=[])
