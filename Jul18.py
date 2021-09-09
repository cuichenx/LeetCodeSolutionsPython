from helpers import print_assert, ListNode, LinkedList as ll


class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if k == 1:
            return head  # do nothing

        cur = head
        prev_link = None
        this_group = []
        while cur is not None:
            this_group.append(cur)
            if len(this_group) == k:
                # swicheroni here
                this_group[0].next = cur.next  # do this in case the rest of the nodes don't constitute a full group
                for i in range(1, k):
                    this_group[i].next = this_group[i - 1]
                if prev_link is not None:
                    prev_link.next = this_group[-1]
                else:
                    # first group. make head point to the new head
                    head = this_group[-1]
                prev_link = this_group[0]
                cur = this_group[0]
                this_group = []

            cur = cur.next

        return head
    # Time: O(n)
    # Space: O(k)


if __name__ == '__main__':
    sol = Solution()
    head = ll.makeLinkedList([1, 2, 3, 4, 5])
    print_assert(actual=ll.printLinkedList(sol.reverseKGroup(head, 2)), expected=[2, 1, 4, 3, 5])
    head = ll.makeLinkedList([1, 2, 3, 4, 5])
    print_assert(actual=ll.printLinkedList(sol.reverseKGroup(head, 3)), expected=[3, 2, 1, 4, 5])
    head = ll.makeLinkedList([1, 2, 3, 4, 5])
    print_assert(actual=ll.printLinkedList(sol.reverseKGroup(head, 1)), expected=[1, 2, 3, 4, 5])
    head = ll.makeLinkedList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print_assert(actual=ll.printLinkedList(sol.reverseKGroup(head, 4)),
                 expected=[4, 3, 2, 1, 8, 7, 6, 5, 9, 10])
