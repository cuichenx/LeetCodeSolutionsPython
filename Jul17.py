from collections import defaultdict
from typing import List
from bisect import bisect_left, bisect_right
from helpers import print_assert

class Solution:
    # 927 Three Equal Parts
    # You are given an array arr which consists of only zeros and ones,
    # divide the array into three non-empty parts such that all of these parts represent the same binary value.
    #
    # If it is possible, return any [i, j] with i + 1 < j, such that:
    #
    # arr[0], arr[1], ..., arr[i] is the first part,
    # arr[i + 1], arr[i + 2], ..., arr[j - 1] is the second part, and
    # arr[j], arr[j + 1], ..., arr[arr.length - 1] is the third part.
    # All three parts have equal binary values.
    # If it is not possible, return [-1, -1].
    #
    # Note that the entire part is used when considering what binary value it represents. For example, [1,1,0]
    # represents 6 in decimal, not 3. Also, leading zeros are allowed, so [0,1,1] and [1,1] represent the same value.
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        # there must be 3k ones in the list.
        number_of_ones = arr.count(1)
        if number_of_ones == 0:
            return [0, 2]  # all zeros
        elif number_of_ones//3 * 3 != number_of_ones:
            return [-1, -1]  # not possible

        # find the first and last ones of the first group. the first group may contain more zeros to the right
        c = 0
        first_one = -1
        i = 0
        while c < number_of_ones//3:
            if arr[i] == 1:
                if first_one == -1:
                    first_one = i
                c += 1
            i += 1

        last_one = i  # just past last one
        len_group = last_one - first_one

        # advance to first one of second group
        zeros_after_first = 0  # number of zeros between first and second group,
        # some of which are trailing zeros of first group, others are leading zeros of second group
        while arr[i] == 0:
            zeros_after_first += 1
            i += 1

        # check whether second group matches first group
        c = 0
        while c < number_of_ones//3:
            if arr[i] != arr[i - len_group - zeros_after_first]:
                return [-1, -1]  # does not match
            if arr[i] == 1:
                c += 1
            i += 1

        zeros_after_second = 0
        while arr[i] == 0:
            zeros_after_second += 1
            i += 1

        # check whether third group matches first group
        c = 0
        while c < number_of_ones//3:
            if arr[i] != arr[i - len_group*2 - zeros_after_first - zeros_after_second]:
                return [-1, -1]
            if arr[i] == 1:
                c += 1
            i += 1

        trailing_zeros = len(arr) - i
        if trailing_zeros <= zeros_after_first and trailing_zeros <= zeros_after_second:
            return [first_one + len_group + trailing_zeros - 1,
                    first_one + len_group*2 + zeros_after_first + trailing_zeros]
        else:
            return [-1, -1]

# time: O(N)
# space: O(1)



# 1348
class TweetCounts:

    def __init__(self):
        self.data = defaultdict(list)  # dictionary where key = tweetname and value = sorted list of arrival times
        self.intervals = {
            'minute': 60,
            'hour': 3600,
            'day': 86400,
        }

    def recordTweet(self, tweetName: str, time: int) -> None:
        idx = bisect_right(self.data[tweetName], time)
        self.data[tweetName].insert(idx, time)

    def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
        interval = self.intervals[freq]
        time_intervals = []
        t = startTime
        while t <= endTime:
            time_intervals.append((t, min(t+interval-1, endTime)))
            t += interval
        ret = []
        data = self.data[tweetName]
        for start, end in time_intervals:
            ret.append(bisect_right(data, end) - bisect_left(data, start))
        return ret

# Your TweetCounts object will be instantiated and called as such:
# obj = TweetCounts()
# obj.recordTweet(tweetName,time)
# param_2 = obj.getTweetCountsPerFrequency(freq,tweetName,startTime,endTime)



if __name__ == '__main__':
    sol = Solution()
    # print_assert(sol.threeEqualParts([1, 0, 1, 0, 1]), [0, 3])
    # print_assert(sol.threeEqualParts([1, 1, 0, 1, 1]), [-1, -1])
    # print_assert(sol.threeEqualParts([1, 1, 0, 0, 1]), [0, -2])
    # print_assert(sol.threeEqualParts([1, 0, 0, 0, 1]), [-1, -1])
    # print_assert(sol.threeEqualParts([0, 0, 0, 0, 0]), [0, 2])
    # print_assert(sol.threeEqualParts([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), [4, 12])

    obj = TweetCounts()
    obj.recordTweet('tweet3', 0)
    obj.recordTweet('tweet3', 60)
    obj.recordTweet('tweet3', 10)
    print_assert(
        obj.getTweetCountsPerFrequency("minute", "tweet3", 0, 59),
        [2])
    print_assert(
        obj.getTweetCountsPerFrequency("minute", "tweet3", 0, 60),
        [2, 1])
    obj.recordTweet("tweet3", 120)
    print_assert(
        obj.getTweetCountsPerFrequency("hour", "tweet3", 0, 210),
        [4])
