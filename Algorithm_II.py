import bisect
import math
import time
from collections import defaultdict
from typing import List, Optional

from helpers import print_assert, ListNode, LinkedList as ll
import heapq


class AlgoIIDay1:
    # 34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        idx1 = bisect.bisect_left(nums, target)
        if idx1 < len(nums) and nums[idx1] == target:
            idx2 = bisect.bisect_right(nums, target)
            return [idx1, idx2 - 1]
        else:
            return [-1, -1]

    def test1(self):
        print_assert(self.searchRange([5, 7, 7, 8, 8, 10], 8), [3, 4])
        print_assert(self.searchRange([5, 7, 7, 8, 8, 10], 6), [-1, -1])
        print_assert(self.searchRange([], 0), [-1, -1])

    # 33. Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        # two binary searches
        # first one to locate the minimum element (orig start)
        # second one to do normal binary search, given adjusted indices
        n = len(nums)
        lo, hi = -1, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if nums[mid] <= nums[hi]:
                # second half is sorted.
                hi = mid
            else:
                # first half is sorted
                lo = mid
        orig_start = hi

        # print(orig_start)
        def f(x):
            # orig index in sorted array to new index in rotated array
            return (x + orig_start) % n

        lo, hi = 0, n  # lo mid hi are rotated indices
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= nums[f(mid)]:
                hi = mid
            else:
                lo = mid + 1
        if nums[f(lo)] == target:
            return f(lo)
        else:
            return -1

    def test2(self):
        print_assert(self.search([0], 0), 0)
        print_assert(self.search([0, 1], 0), 0)
        print_assert(self.search([0, 1, 2], 0), 0)
        print_assert(self.search([0, 1, 2, 3], 0), 0)
        print_assert(self.search([0, 1, 2, 3, 4], 0), 0)
        print_assert(self.search([4, 5, 6, 7, 0, 1, 2], 3), -1)
        print_assert(self.search([3, 1], 1), 1)
        print_assert(self.search([1, 3], 1), 0)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], 3), 10)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4], 3), 9)
        print_assert(self.search([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], 3), 8)
        print_assert(self.search([5, 6, 7, 8, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 8, 0, 1, 2, 3, 4], 3), 7)
        print_assert(self.search([5, 6, 7, 0, 1, 2, 3, 4], 7), 2)
        print_assert(self.search([5, 6, 7, 0, 1, 2, 3, 4], 3), 6)
        print_assert(self.search([5, 6, 0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([5, 6, 0, 1, 2, 3, 4], 3), 5)
        print_assert(self.search([5, 0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([5, 0, 1, 2, 3, 4], 3), 4)
        print_assert(self.search([0, 1, 2, 3, 4], 7), -1)
        print_assert(self.search([0, 1, 2, 3, 4], 3), 3)

    # 74. Search a 2D Matrix
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])

        lo, hi = 0, m*n  # lo mid hi are rotated indices
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= matrix[mid//n][mid%n]:
                hi = mid
            else:
                lo = mid + 1
        return lo < m*n and matrix[lo//n][lo%n] == target

    def test3(self):
        print_assert(self.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3), True)
        print_assert(self.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13), False)
        print_assert(self.searchMatrix([[1]], 1), True)
        print_assert(self.searchMatrix([[1]], 2), False)


class AlgoIIDay2:
    def findMin(self, nums: List[int]) -> int:
        # similar to yesterday's question 2??
        n = len(nums)
        lo, hi = -1, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if nums[mid] <= nums[hi]:
                # second half is sorted.
                hi = mid
            else:
                # first half is sorted
                lo = mid
        return nums[hi]

    def test1(self):
        print_assert(self.findMin([3,4,5,1,2]), 1)
        print_assert(self.findMin([4,5,6,7,0,1,2]), 0)
        print_assert(self.findMin([11,13,15,17]), 11)
        print_assert(self.findMin([1,11,13,15,17]), 1)

    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        # if lo > mid, then first half must contain a peak, or nums[0] is a peak
        # elif mid < hi, then second half must contain a peak, or nums[n-1] is a peak
        # else, (lo < mid and mid > hi), check go to the higher one of mid-1 and mid+1
        lo, hi = 0, n-1
        while lo < hi - 1:
            mid = (lo+hi)//2
            if nums[mid] > nums[mid-1] and nums[mid] > nums[mid+1]:
                return mid
            if nums[lo] > nums[mid]:
                hi = mid
            elif nums[hi] > nums[mid]:
                lo = mid
            elif nums[mid-1] > nums[mid]:
                hi = mid
            else:
                lo = mid
        return lo if nums[lo]>nums[hi] else hi

    def test2(self):
        print_assert(self.findPeakElement([1, 2, 3, 1]), 2)
        print_assert(self.findPeakElement([1, 2, 1, 3, 5, 6, 4]), (5, 1))
        print_assert(self.findPeakElement([1, 2, 3, 4, 5, 6, 7, 8, 7]), 7)
        print_assert(self.findPeakElement([1, 2, 3, 4, 5, 6, 7, 8, 9]), 8)
        print_assert(self.findPeakElement([3, 2, 1]), 0)
        print_assert(self.findPeakElement([3, 4, 2, 1]), 1)


class AlgoIIDay3:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # fist handle the case where first few elements are duplicates
        while True:
            first_elements_duplicates = False
            while head and head.next and head.val == head.next.val:
                head = head.next
                first_elements_duplicates = True
            if first_elements_duplicates:
                head = head.next  # and continue. the next head might also be duplicates
            else:
                break
        # now first few elements aren't duplicates

        keep = None
        cur = head
        while cur:
            if keep is None:
                # looking for duplicate nodes
                if not cur.next or not cur.next.next or cur.next.val != cur.next.next.val:  # no duplicate
                    cur = cur.next
                else:
                    # duplicate incoming, keep track of cur, to be linked to a later node.
                    keep = cur
            else:
                # looking at duplicate nodes here
                if (cur.next and cur.val == cur.next.val) or \
                        (cur.next and cur.next.next and cur.next.val == cur.next.next.val):
                    cur = cur.next   # still in a run of duplicates, keep advancing
                else:
                    # run of duplicates ended
                    keep.next = cur.next
                    keep = None
                    cur = cur.next
        if keep is not None:
            # if list finished while looking for more duplicates, then modified list ends with keep.
            keep.next = None
        return head

    def test1(self):
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 2, 3, 3, 4, 4, 5]))), [1, 2, 5])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 1, 1, 2, 3]))), [2, 3])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 3, 3, 4, 4, 4, 4, 4, 5, 7, 7, 8, 9]))), [1, 5, 8, 9])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 2, 3, 4]))), [1, 2, 3, 4])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 1, 2, 3, 4]))), [2, 3, 4])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([]))), [])
        print_assert(ll.printLinkedList(self.deleteDuplicates(ll.makeLinkedList([1, 1, 2, 2]))), [])

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ret = []
        nums.sort()
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue  # don't calculate duplicate i
            k = n - 1
            j = i + 1
            while j < k:
                cur_sum = nums[i] + nums[j] + nums[k]
                if cur_sum > 0:
                    k -= 1
                elif cur_sum < 0:
                    j += 1
                else:
                    ret.append([nums[i], nums[j], nums[k]])
                    # increment j until it's different. decrement k until it's different
                    j += 1
                    while j < k and nums[j-1] == nums[j]: j += 1
                    k -= 1
                    while k > i and nums[k+1] == nums[k]: k -= 1
        return ret

    def test2(self):
        print_assert(self.threeSum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]])
        print_assert(self.threeSum([]), [])
        print_assert(self.threeSum([0]), [])
        print_assert(self.threeSum([0, 0, 0]), [[0, 0, 0]])
        print_assert(self.threeSum([0, 0, 0, 0]), [[0, 0, 0]])
        print_assert(self.threeSum([0, 0, 0, 0, -1]), [[0, 0, 0]])
        print_assert(self.threeSum([0, 0, 0, 0, -1, 1]), [[-1, 0, 1], [0, 0, 0]])
        print_assert(self.threeSum([-2, 0, 1, 1, 2]), [[-2, 0, 2], [-2, 1, 1]])

if __name__ == '__main__':
    AlgoIIDay3().test2()
