from typing import Tuple, Optional

from helpers import print_assert, TreeNode, Tree as tr


class Solution:
    # 235. Lowest Common Ancestor of a Binary Search Tree
    # Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
    def lowestCommonAncestor_slow(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        return self.isDescendentOf(root, p, q)[-1]

    def isDescendentOf(self, root: TreeNode, target1: TreeNode, target2: TreeNode) -> Tuple[bool, bool, Optional[TreeNode]]:
        if root is None:
            return False, False, None

        lfound1, lfound2, lres = self.isDescendentOf(root.left, target1, target2)
        if lres:
            # result already found on the left sub tree
            return True, True, lres
        if (lfound1 and root == target2) or (lfound2 and root == target1):
            # current node is target1 or target2, done
            return True, True, root

        rfound1, rfound2, rres = self.isDescendentOf(root.right, target1, target2)
        if rres:
            return True, True, rres
        if (rfound1 and root == target2) or (rfound2 and root == target1):
            # current node is target1 or target2, done
            return True, True, root

        # finally, if one is on the left and one on the right, then current root is lowest common ancester
        if (lfound1 and rfound2) or (lfound2 and rfound1):
            return True, True, root

        return (lfound1 or rfound1 or root == target1), (lfound2 or rfound2 or root == target2), None

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root



if __name__ == '__main__':
    null = None
    sol = Solution()
    nodes = tr.list2listOfNodes([6, 2, 8, 0, 4, 7, 9, null, null, 3, 5])
    print_assert(actual=sol.lowestCommonAncestor(nodes[0], nodes[1], nodes[2]),
                 expected=nodes[0])
    print_assert(actual=sol.lowestCommonAncestor(nodes[0], nodes[1], nodes[4]),
                 expected=nodes[1])
    print_assert(actual=sol.lowestCommonAncestor(nodes[0], nodes[9], nodes[10]),
                 expected=nodes[4])
    print_assert(actual=sol.lowestCommonAncestor(nodes[0], nodes[9], nodes[6]),
                 expected=nodes[0])
