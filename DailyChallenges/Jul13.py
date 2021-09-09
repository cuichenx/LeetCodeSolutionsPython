from typing import List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # 162
    def findPeakElement(self, nums: List[int]) -> int:
        return self.findPeak(nums, 0, len(nums)-1)

    def findPeak(self, nums: List[int], beg, end) -> int:
        n = len(nums)
        if end == 0:
            return 0
        if end == 1:
            return 0 if nums[0] > nums[1] else 1
        if beg >= n - 2:
            return n-1 if nums[-1] > nums[-2] else n-2
        mid = (beg + end) // 2
        if nums[mid] > nums[mid-1] and nums[mid] > nums[mid+1]:
            return mid

        if nums[mid-1] > nums[mid]:
            # check left half
            return self.findPeak(nums, beg, mid)
        else:
            return self.findPeak(nums, mid, end)

    # 2
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        res_head = ListNode()
        res_cur = res_head

        carry = 0
        first = True
        while l1 is not None or l2 is not None or carry > 0:
            if not first:  # starting from the second iter
                res_cur.next = ListNode()
                res_cur = res_cur.next
            first = False

            l1val = 0 if l1 is None else l1.val
            l2val = 0 if l2 is None else l2.val
            s = l1val + l2val + carry
            res_cur.val = s%10
            carry = int(s >= 10)
            if l1 is not None: l1 = l1.next
            if l2 is not None: l2 = l2.next

        return res_head

    def makeLinkedList(self, l: List[int]) -> ListNode:
        node_head = ListNode()
        node_cur = node_head
        for i in range(len(l)-1):
            node_cur.val = l[i]
            node_cur.next = ListNode()
            node_cur = node_cur.next
        node_cur.val = l[-1]
        return node_head

    def printLinkedList(self, l: ListNode) -> List:
        lst = []
        while True:
            lst.append(l.val)
            if l.next:
                l = l.next
            else: break

        return lst



if __name__ == '__main__':
    sol = Solution()
    # print(sol.findPeakElement([1,2,3]), (2))
    # print(sol.findPeakElement([1,2,1,3,5,6,4]), (1, 5))
    # print(sol.findPeakElement([1]), (0,))
    # print(sol.findPeakElement([1,2]), (1,))
    # print(sol.findPeakElement([1,2,1]), (1,))
    # print(sol.findPeakElement([3, 0, 3]), (0,2))
    # print(sol.findPeakElement([9, 5, 3, 2, 1, 0, -1]), (0,))
    # print(sol.findPeakElement([1,2, 3, 4, 5, 6, 7]), (6,))
    # print(sol.findPeakElement([-1, 4, 3, 7, 8, 13, -2, -4, 7]), (1, 5, 8))


    # s = sol.addTwoNumbers(
    #     sol.makeLinkedList([9,9,9,9,9,9,9]),
    #     sol.makeLinkedList([9,9,9,9]))
    # print(sol.printLinkedList(s), [8,9,9,9,0,0,0,1])
    # s = sol.addTwoNumbers(
    #     sol.makeLinkedList([2, 4, 3]),
    #     sol.makeLinkedList([5, 6, 4]))
    # print(sol.printLinkedList(s), [7, 0, 8])
    s = sol.addTwoNumbers(
        sol.makeLinkedList([1, 2]),
        sol.makeLinkedList([0]))
    print(sol.printLinkedList(s), [0])