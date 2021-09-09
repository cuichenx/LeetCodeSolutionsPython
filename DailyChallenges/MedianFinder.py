from bisect import bisect_left

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.cur_med_idx = -0.5

    def addNum(self, num: int) -> None:
        i = bisect_left(self.data, num)
        self.data.insert(i, num)
        self.cur_med_idx += 0.5  # index of median is always half way

    def findMedian(self) -> float:
        if self.cur_med_idx == int(self.cur_med_idx):
            return self.data[int(self.cur_med_idx)]
        else:
            return (self.data[int(self.cur_med_idx+0.5)] + self.data[int(self.cur_med_idx-0.5)])/2

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
if __name__ == '__main__':
    medianFinder = MedianFinder()
    medianFinder.addNum(-1)
    print(medianFinder.findMedian())
    medianFinder.addNum(-2)
    print(medianFinder.findMedian())
    medianFinder.addNum(-3)
    print(medianFinder.findMedian())
    medianFinder.addNum(-4)
    print(medianFinder.findMedian())
    medianFinder.addNum(-5)
    print(medianFinder.findMedian())
