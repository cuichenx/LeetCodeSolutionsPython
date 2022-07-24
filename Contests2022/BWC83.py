import time
from itertools import product
from typing import Optional, List

from helpers import print_assert, print_assert_stream
from bisect import bisect_left, bisect_right
import heapq
from collections import Counter, defaultdict


class Solution:
    def bestHand(self, ranks: List[int], suits: List[str]) -> str:
        if suits[0] == suits[1] == suits[2] == suits[3] == suits[4]:
            return 'Flush'
        count = Counter(ranks).most_common(1)[0][1]
        if count >= 3:
            return 'Three of a Kind'
        elif count == 2:
            return 'Pair'
        else:
            return 'High Card'

    def test1(self):
        print_assert(self.bestHand(ranks = [13,2,3,1,9], suits = ["a","a","a","a","a"]), 'Flush')
        print_assert(self.bestHand(ranks = [4,4,2,4,4], suits = ["d","a","a","b","c"]), 'Three of a Kind')
        print_assert(self.bestHand(ranks = [10,10,2,12,9], suits = ["a","b","c","a","d"]), 'Pair')
        print_assert(self.bestHand(ranks = [10,11,2,12,9], suits = ["a","b","c","a","d"]), 'High Card')

    def zeroFilledSubarray(self, nums: List[int]) -> int:
        def triangle(x):
            return x*(x+1)/2

        tot = 0
        cur_run = 0
        for num in nums:
            if num == 0:
                cur_run += 1
            else:
                tot += triangle(cur_run)
                cur_run = 0
        tot += triangle(cur_run)
        return int(tot)

    def test2(self):
        print_assert(self.zeroFilledSubarray([1,3,0,0,2,0,0,4]), 6)
        print_assert(self.zeroFilledSubarray([0,0,0,2,0,0]), 9)
        print_assert(self.zeroFilledSubarray([2,10,2019]), 0)

    def test3(self):
        nc = NumberContainers()
        print_assert(nc.find(10), -1)
        nc.change(2, 10)
        nc.change(1, 10)
        nc.change(3, 10)
        nc.change(5, 10)
        print_assert(nc.find(10), 1)
        nc.change(1, 20)
        print_assert(nc.find(10), 2)
        nc.change(2, 20)
        print_assert(nc.find(10), 3)
        nc.change(3, 20)
        print_assert(nc.find(10), 5)
        nc.change(5, 20)
        print_assert(nc.find(10), -1)
        null = None
        print_assert_stream(nc, ["NumberContainers","change","change","change","find","change","find","find","change","find","find","find","change","find","change","change","find","find","change","find","change","change","find","find","find","change","find","find","change","find","find","find","change","find","change","change","find","change","change","change","change","find","change","change","find","change","find","find","change","change","find","change","find","change","change","find","change","change","find","change","find","change","find","find","find","change","find","find","change","change","find","change","change","find","change","change","find","find","change","find","change","change","find","find","find","find","change","find","change","find","change","change","find","change","find","change","change","change","change","find","find","find","change","change","change","change","change","find","change","change","change","find","change","change","find","change","change","find","change","change","change","change","change","change","find","find","find","find","change","find","find","find","find","change","find","change","find","find","find","find","change","find","find","change","find","find","change","change","change","change","change","change","find","find","change","change","change","change","find","change","change","change","change","change","change","change","change","find","change","find","change","find","change","change","change","change","find","change","find","change","change","change","change","change","change","change","change","change","change","find","change","change","find","change","change","change","find","find","change","find","find"],
[[],[158,9],[75,85],[75,187],[77],[109,113],[184],[77],[17,191],[113],[35],[184],[164,119],[9],[19,151],[142,50],[77],[85],[35,164],[184],[118,164],[3,164],[184],[113],[135],[72,105],[9],[187],[34,105],[135],[164],[135],[20,164],[187],[158,184],[44,50],[191],[164,50],[20,191],[158,191],[107,113],[187],[158,50],[142,9],[151],[35,119],[113],[105],[127,77],[164,164],[187],[72,191],[113],[132,164],[7,9],[85],[71,187],[7,187],[9],[20,185],[35],[7,151],[119],[135],[77],[155,187],[164],[135],[183,151],[110,164],[50],[20,85],[19,9],[85],[175,9],[116,105],[187],[164],[107,35],[185],[147,184],[109,184],[35],[184],[187],[113],[178,85],[9],[178,151],[85],[107,164],[116,135],[113],[107,164],[77],[116,35],[172,35],[200,187],[142,50],[50],[187],[105],[127,9],[34,164],[178,135],[183,50],[34,35],[184],[147,77],[172,35],[132,151],[119],[7,185],[109,185],[187],[110,135],[175,35],[35],[127,187],[71,164],[188,9],[35,50],[107,191],[158,119],[85],[50],[35],[77],[183,164],[119],[9],[77],[50],[164,9],[151],[172,77],[50],[135],[113],[77],[200,9],[77],[184],[142,105],[119],[9],[75,185],[142,113],[127,119],[110,85],[7,135],[127,185],[185],[77],[200,50],[164,164],[19,35],[172,113],[135],[178,35],[72,35],[142,85],[3,113],[109,151],[110,77],[35,119],[75,164],[105],[142,113],[164],[127,105],[119],[110,135],[158,35],[35,164],[35,9],[135],[178,50],[119],[73,185],[19,85],[155,151],[44,187],[116,191],[158,35],[110,191],[72,187],[7,9],[17,135],[35],[200,185],[142,185],[164],[175,187],[188,185],[172,50],[9],[119],[110,191],[35],[35]],
                            [null,null,null,null,-1,null,-1,-1,null,109,-1,-1,null,158,null,null,-1,-1,null,-1,null,null,-1,109,-1,null,158,75,null,-1,3,-1,null,75,null,null,17,null,null,null,null,75,null,null,19,null,107,34,null,null,75,null,107,null,null,-1,null,null,142,null,-1,null,35,-1,127,null,3,-1,null,null,44,null,null,20,null,null,71,3,null,-1,null,null,107,109,71,-1,null,19,null,20,null,null,-1,null,127,null,null,null,null,44,71,34,null,null,null,null,null,109,null,null,null,35,null,null,71,null,null,34,null,null,null,null,null,null,20,35,34,147,null,158,19,147,35,null,132,null,35,110,-1,147,null,147,-1,null,158,19,null,null,null,null,null,null,75,147,null,null,null,null,7,null,null,null,null,null,null,null,null,-1,null,71,null,35,null,null,null,null,7,null,-1,null,null,null,null,null,null,null,null,null,null,34,null,null,71,null,null,null,7,-1,null,34,34]
                            )

    def shortestSequence_slow(self, rolls: List[int], k: int) -> int:
        # build inverted index
        inverted_index = {roll+1: [] for roll in range(k)}
        for i, roll in enumerate(rolls):
            inverted_index[roll].append(i)

        seq_to_pos_idx = {}

        for seq_len in range(1, len(rolls)+1):
            t0 = time.time()
            for seq in product(* [range(1, k+1)]*seq_len ):
                pos_idx = -1
                for seq_i in seq:
                    one_larger = bisect_right(inverted_index[seq_i], pos_idx)
                    if one_larger == len(inverted_index[seq_i]):
                        print(seq_len, 'partial', time.time() - t0)
                        return seq_len  # found a sequence that cannot be completed!!!
                    else:
                        pos_idx = inverted_index[seq_i][one_larger]

                    # for index in inverted_index[seq_i]:
                    #     if index > pos_idx:
                    #         pos_idx = index
                    #         break
                    # else:  # no break
                    #     return seq_len  # found a sequence that cannot be completed!!!
            print(seq_len, time.time()-t0)

    def shortestSequence_stillslow(self, rolls: List[int], k: int) -> int:
        # build inverted index
        inverted_index = {roll + 1: [] for roll in range(k)}
        for i, roll in enumerate(rolls):
            inverted_index[roll].append(i)
        seq_to_pos_idx = {tuple(): -1}
        for seq_len in range(1, len(rolls) + 1):
            t0 = time.time()
            seq_to_pos_idx_prev = seq_to_pos_idx
            seq_to_pos_idx = {}

            for seq_one_short, pos_idx_one_short in seq_to_pos_idx_prev.items():
                for last_roll in range(1, k+1):
                    one_larger = bisect_right(inverted_index[last_roll], pos_idx_one_short)
                    if one_larger == len(inverted_index[last_roll]):
                        print(seq_len, 'partial', time.time() - t0)
                        return seq_len  # found a sequence that cannot be completed!!!
                    else:
                        pos_idx = inverted_index[last_roll][one_larger]

                    seq = list(seq_one_short) + [last_roll]
                    seq_to_pos_idx[tuple(seq)] = pos_idx
            print(seq_len, time.time()-t0)

    def shortestSequence(self, rolls: List[int], k: int) -> int:
        # didn't think of this during the contest :(
        # idea: find non overlapping subarrays that contains all of 1..k
        # https://leetcode.com/problems/shortest-impossible-sequence-of-rolls/discuss/2322321/Count-Subarrays-with-All-Dice
        # only way to form all sequences is to select one from each subarray
        # e.g. 2,1,4,2,1,1,2,2,2,3, || 2,1,4,2,4,2,2,1,1,4,2,4,3, || 2,3,4,1, ||| 3,4,2,1,|| 1,2,3  => return 5
        seq_len = 1
        seen = [False] * (k+1)  # first elem is dummy
        num_seen = 0
        for roll in rolls:
            if not seen[roll]:
                seen[roll] = True
                num_seen += 1
            if num_seen == k:
                seq_len += 1
                seen = [False] * (k + 1)  # first elem is dummy
                num_seen = 0
        if num_seen == k:
            seq_len += 1
        return seq_len

    def test4(self):
        print_assert(self.shortestSequence([4,2,1,2,3,3,2,4,1], 4), 3)
        print_assert(self.shortestSequence([1,1,2,2], 2), 2)
        print_assert(self.shortestSequence([1,1,3,2,2,2,3,3], 4), 1)
        print_assert(self.shortestSequence([2, 1], 2), 2)
        print_assert(self.shortestSequence([2,1,4,2,1,1,2,2,2,3,2,1,4,2,4,2,2,1,1,4,2,4,3,2,3,4,1,3,4,2,1,1,2,3,1,4,2,2,3,4,1,2,1,1,1,1,1,4,3,2,3,4,1,4,1,3,3,2,1,4,3,4,2,3,2], 4), 11)
        print_assert(self.shortestSequence([2,2,2,2,2,2,1,2,2,2,1,1,1,2,2,2,2,1,2,1,1,2,2,2,2,1,1,1,1,2,1,1,2,1,1,2,2,1,1,1,2,1,1,1,2,2,1,2,1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,2,2,1,2,1,2,1,2,2,1,1,2,1,1,1,1,2,2,2,2,1,2,1,1,2,1,2,1,1,2,2,1,2,1,1,2,2,2,1,2,2,1,1,2,2,1,2,1,1,2,1,1,1,1,2,2,1,2,2,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,1,1,2,1,1,1,1,2,1,1,2,2,2,2,2,2,2,1,2,2,2,2,2,2,1,2,1,1,2,1,2,1,2,2,2,2,2,2,1,1,2,1,2,2,2,2,1,2,1,2,1,2,1,1,1,2,1,1,1,2,1,1,2,2,1,1,1,1,2,2,2,2,1,2,1,1,1,1,1,2,1,1,1,1,2,2,1,1,1,2,2,1,2,1,2,1,1,2,2,2,1,1,2,1,2,1,2,2,1,1,1,1,2,2,2,1,1,2,2,1,1,1,1,1,1,2,1,1,2,2,2,1,1,2,1,2,2,2,2,2],2), 93)


class NumberContainers:

    def __init__(self):
        self.idx_to_num = {}
        self.num_to_idx = defaultdict(list)

    def change(self, index: int, number: int) -> None:
        cur_num = self.idx_to_num.get(index, -1)
        self.idx_to_num[index] = number
        heapq.heappush(self.num_to_idx[number], index)

        # if cur_num > -1:
        #     # cur_num does not exist at index anymore
        #     self.num_to_idx[cur_num].remove(index)
        #     heapq.heapify(self.num_to_idx[cur_num])

    def find(self, number: int) -> int:
        indices = self.num_to_idx.get(number, [])
        if len(indices) == 0:
            return -1

        # make sure number still exists at min idx
        while (self.idx_to_num[indices[0]] != number):
            heapq.heappop(indices)
            if len(indices) == 0:
                return -1
        return indices[0]

# Your NumberContainers object will be instantiated and called as such:
# obj = NumberContainers()

# obj.change(index,number)
# param_2 = obj.find(number)


if __name__ == '__main__':
    Solution().test4()

