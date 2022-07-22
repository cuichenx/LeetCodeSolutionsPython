from collections import deque
from typing import List, Optional

def print_assert(actual, expected):
    if isinstance(expected, tuple):  # multiple acceptable answers
        is_correct = actual in expected
    else:
        is_correct = actual == expected

    print(f"{'✔️' if is_correct else '❌️'} Actual: {actual}   Expected: {expected}")
    return is_correct

def print_assert_stream(obj, ops, inputs, expected):
    for i, (op, inp, ex) in enumerate(zip(ops, inputs, expected)):
        if op == obj.__class__.__name__:
            print(i, "✔ Initializer.")
            continue
        print(i, end=' ')
        if not print_assert(obj.__getattribute__(op)(*inp), ex):
            break
    else:  # nobreak
        print("All tests passed")
        return True
    print("Some tests failed, stopping")
    return False

def print_matrix(matrix):
    # https://stackoverflow.com/questions/13214809/pretty-print-2d-list
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

class Bisect:
    @classmethod
    def bisect_right(cls, a, x, lo=0, hi=None):
        """Return the index where to insert item x in list a, assuming a is sorted.

        The return value i is such that all e in a[:i] have e <= x, and all e in
        a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
        insert just after the rightmost x already there.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    @classmethod
    def bisect_left(cls, a, x, lo=0, hi=None):
        """Return the index where to insert item x in list a, assuming a is sorted.

        The return value i is such that all e in a[:i] have e < x, and all e in
        a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
        insert just before the leftmost x already there.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x <= a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode(val={self.val}, next={None if not self.next else self.next.val})"


class LinkedList:
    @classmethod
    def makeLinkedList(cls, l: List[int]) -> Optional[ListNode]:
        if not l:
            return None
        node_head = ListNode()
        node_cur = node_head
        for i in range(len(l) - 1):
            node_cur.val = l[i]
            node_cur.next = ListNode()
            node_cur = node_cur.next
        node_cur.val = l[-1]
        return node_head

    @classmethod
    def printLinkedList(cls, l: Optional[ListNode]) -> List:
        lst = []
        while l:
            lst.append(l.val)
            l = l.next
        return lst


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        l = None if self.left is None else self.left.val
        r = None if self.right is None else self.right.val
        return f"TreeNode(val={self.val}, l={l}, r={r})"


class Tree:
    @classmethod
    def list2tree(cls, l: List[Optional[int]]) -> Optional[TreeNode]:
        if not l:
            return None
        root = TreeNode(l[0])
        q = [root]
        cur_elem = 1
        while len(q) != 0 and cur_elem < len(l):
            node = q.pop(0)
            if l[cur_elem] is not None:
                node.left = TreeNode(l[cur_elem])
                q.append(node.left)
            cur_elem += 1
            if cur_elem < len(l) and l[cur_elem] is not None:
                node.right = TreeNode(l[cur_elem])
                q.append(node.right)
            cur_elem += 1
        return root

    @classmethod
    def list2listOfNodes(cls, l: List[Optional[int]]) -> List[TreeNode]:
        root = TreeNode(l[0])
        ret = [root]
        q = [root]
        cur_elem = 1
        while len(q) != 0 and cur_elem < len(l):
            node = q.pop(0)
            if l[cur_elem] is not None:
                node.left = TreeNode(l[cur_elem])
                q.append(node.left)
            ret.append(node.left)
            cur_elem += 1
            if cur_elem < len(l) and l[cur_elem] is not None:
                node.right = TreeNode(l[cur_elem])
                q.append(node.right)
            ret.append(node.right)
            cur_elem += 1
        return ret

    @classmethod
    def tree2list(cls, root: TreeNode) -> List[int]:
        ''' BFS to print the content of a tree to a list '''
        q = deque([root])
        ret = []

        while len(q) > 0:
            cur = q.popleft()
            ret.append(cur.val if cur is not None else None)
            if cur is not None:
                q.append(cur.left)
                q.append(cur.right)

        # drop trailing None's
        while len(ret) > 0 and ret[-1] is None:
            ret.pop(-1)
        return ret


# N-ary tree
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

    def __repr__(self):
        children = None if self.children is None else [c.val for c in self.children]
        return f"Node(val={self.val}, c={children})"


class NaryTree:
    # Nary-Tree input serialization is represented in their level order traversal,
    # each group of children is separated by the null value (See examples in Q429).
    @classmethod
    def list2tree(cls, l: List[Optional[int]]) -> Optional[Node]:
        if len(l) < 1:
            return None
        root = Node(val=l[0])
        cur_row = [root]
        idx = 2  # [root, null, ...]
        while idx < len(l):
            parents = cur_row
            cur_row = []
            for p in parents:
                while idx < len(l) and l[idx] is not None:
                    new_node = Node(l[idx])
                    if p.children is None:
                        p.children = [new_node]
                    else:
                        p.children.append(new_node)
                    cur_row.append(new_node)
                    # l[idx] is None at this point. pass to next
                    idx += 1
                # this parent has finished, go to next parent
                idx += 1

        return root



if __name__ == '__main__':
    null = None
    # root = NaryTree.list2tree([1, null, 3, 2, 4, null, 5, 6])
    root = NaryTree.list2tree([1, null, 2, 3, 4, 5, null, null, 6, 7, null, 8, null, 9, 10,
                               null, null, 11, null, 12, null, 13, null, null, 14])
    stop = 0
