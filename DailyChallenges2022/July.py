from typing import Optional, List

from helpers import print_assert, ListNode, LinkedList as ll
from bisect import bisect_left, bisect_right
import heapq

class Q92:
    # 92. Reverse Linked List II
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # suppose there are regions A B C
        # A: before left, B: between left and right, C: after right
        idx = 1
        cur_node = head

        while idx < left - 1 :
            cur_node = cur_node.next
            idx += 1

        if left > 1:
            node_before_left = cur_node
            node_at_left = cur_node = cur_node.next
            idx += 1
        else:
            node_at_left = cur_node

        # now we are at the first node to be swapped
        next_node = cur_node.next
        while idx < right:
            prev_node = cur_node
            cur_node = next_node
            next_node = cur_node.next

            cur_node.next = prev_node
            idx += 1

        # now: we are at the last node to be swapped
        # B left -> C left
        node_at_left.next = next_node
        if left > 1:
            # A right -> B right
            node_before_left.next = cur_node
            return head
        else:
            # there is no "node before left".
            # node before left is actually node at left
            # need to return node at right
            return cur_node

    def test(self):
        # normal case
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3, 4, 5]), 2, 4)
        print_assert(ll.printLinkedList(linked), [1, 4, 3, 2, 5])
        # len(A) == 0
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3, 4, 5]), 1, 4)
        print_assert(ll.printLinkedList(linked), [4, 3, 2, 1, 5])
        # len(B) == 2
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3, 4, 5]), 2, 3)
        print_assert(ll.printLinkedList(linked), [1, 3, 2, 4, 5])
        # len(B) == 1
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3, 4, 5]), 2, 2)
        print_assert(ll.printLinkedList(linked), [1, 2, 3, 4, 5])
        # len(C) == 0
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3, 4, 5]), 3, 5)
        print_assert(ll.printLinkedList(linked), [1, 2, 5, 4, 3])
        # len(ABC) == 3
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3]), 1, 3)
        print_assert(ll.printLinkedList(linked), [3, 2, 1])
        linked = self.reverseBetween(ll.makeLinkedList([1, 2, 3]), 2, 3)
        print_assert(ll.printLinkedList(linked), [1, 3, 2])
        # len(ABC) == 2
        linked = self.reverseBetween(ll.makeLinkedList([1, 2]), 1, 2)
        print_assert(ll.printLinkedList(linked), [2, 1])
        linked = self.reverseBetween(ll.makeLinkedList([1, 2]), 1, 1)
        print_assert(ll.printLinkedList(linked), [1, 2])
        # len(ABC) == 1
        linked = self.reverseBetween(ll.makeLinkedList([1]), 1, 1)
        print_assert(ll.printLinkedList(linked), [1])

class Q86:
    # 86. Partition List
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        lt_end, lt_head, ge_end, ge_head = None, None, None, None
        cur_node = head
        while cur_node:
            if cur_node.val < x:
                if lt_end is None:
                    lt_head = lt_end = cur_node
                else:
                    lt_end.next = cur_node
                    lt_end = lt_end.next
            else:
                if ge_end is None:
                    ge_head = ge_end = cur_node
                else:
                    ge_end.next = cur_node
                    ge_end = ge_end.next
            cur_node = cur_node.next

        if not lt_end:
            return ge_head
        lt_end.next = ge_head
        if ge_end:
            ge_end.next = None
        return lt_head

    def test(self):
        linked = self.partition(ll.makeLinkedList([1, 4, 3, 2, 5, 2]), 3)
        print_assert(ll.printLinkedList(linked), [1, 2, 2, 4, 3, 5])
        linked = self.partition(ll.makeLinkedList([2, 1]), 2)
        print_assert(ll.printLinkedList(linked), [1, 2])
        linked = self.partition(ll.makeLinkedList([]), 2)
        print_assert(ll.printLinkedList(linked), [])
        linked = self.partition(ll.makeLinkedList([3]), 2)
        print_assert(ll.printLinkedList(linked), [3])
        linked = self.partition(ll.makeLinkedList([1, 4, 3, 2, 5, 2]), 10)
        print_assert(ll.printLinkedList(linked), [1, 4, 3, 2, 5, 2])
        linked = self.partition(ll.makeLinkedList([1, 4, 3, 2, 5, 2]), -3)
        print_assert(ll.printLinkedList(linked), [1, 4, 3, 2, 5, 2])

class Q315:
    # 315. Count of Smaller Numbers After Self
    def countSmaller(self, nums: List[int]) -> List[int]:
        # from the right, iteratively insert elements into an array using binary search
        # the index to insert is the number in counts
        arr = []
        counts = []  # actually the reverse of counts

        for num in reversed(nums):
            counts.append(bisect_left(arr, num))
            arr.insert(counts[-1], num)
        return counts[::-1]
    # this is actually quadratic time, although it doesn't TLE
    # there's probably a better solution

    # just a reminder.
    def bisect_left(self, a, x):
        lo, hi = 0, len(a)

        while (lo < hi):
            mid = (lo + hi) // 2
            if x <= a[mid]:
                hi = mid
            else:
                lo = mid + 1

        return lo

    def test(self):
        print_assert(self.countSmaller([5, 2, 6, 1]), [2, 1, 1, 0])
        print_assert(self.countSmaller([-1]), [0])
        print_assert(self.countSmaller([-1, -1]), [0, 0])

    # a = [1, 4, 5, 7], x = 2
    # lo  0 0 0 1
    # hi  4 2 1 1
    # mid 2 1 0

class Q240:
    # 240. Search a 2D Matrix II
    def searchMatrix_slow(self, matrix: List[List[int]], target: int) -> bool:
        def search_block(i0, i1, j0, j1):
            #i0, i1, j0, j1 are all inclusive indices
            if i0 > i1 or j0 > j1: return False
            # print(i0, i1, j0, j1)
            # search in the middle row
            # the point where the middle rows goes from under to over marks the vertical separation
            # then, search bottom left and top right blocks
            if i0 == i1:
                # base case, only one row left. just linear search
                return target in matrix[i0]

            i_mid = (i0 + i1) // 2
            mid_row = matrix[i_mid]
            j_switch = j1+1
            for j in range(j0, j1+1):
                # we could have binary searched this partition point also
                if mid_row[j] == target:
                    return True
                if mid_row[j] > target:
                    j_switch = j
                    break
            exists_bottom_left = search_block(i_mid+1, i1, j0, j_switch-1)
            if exists_bottom_left: return True
            exists_top_right = search_block(i0, i_mid-1, j_switch, j1)
            return exists_top_right

        m, n = len(matrix), len(matrix[0])
        return search_block(0, m-1, 0, n-1)

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # start from bottom left
        # iteratively prune either: everything above it (if target is larger),
        # or: everything right of it (if target is smaller)
        i, j = len(matrix)-1, 0
        while i >= 0 and j < len(matrix[0]):
            if target > matrix[i][j]:
                j += 1
            elif target < matrix[i][j]:
                i -= 1
            else:
                return True
        return False

    def test(self):
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 5), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 8), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 12), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 19), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 2), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 30), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 1), True)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 20), False)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 25), False)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 27), False)
        print_assert(self.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], -1), False)



if __name__ == '__main__':
    Q240().test()