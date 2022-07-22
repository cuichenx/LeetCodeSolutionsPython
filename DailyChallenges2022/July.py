from typing import Optional

from helpers import print_assert, ListNode, LinkedList as ll


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

if __name__ == '__main__':
    Q86().test()