import json


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def printList(l: ListNode):
    while l:
        print("%d, " %(l.val), end = '')
        l = l.next
    print('')


def stringToListNode(input):
    # Generate list from the input
    numbers = json.loads(input)

    # Now convert that list into linked list
    dummyRoot = ListNode(0)
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)
        ptr = ptr.next

    ptr = dummyRoot.next
    return ptr

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.right = None


def prettyPrintTree(node, prefix="", isLeft=True):
    if not node:
        print("Empty Tree")
        return

    if node.right:
        prettyPrintTree(node.right, prefix + ("│   " if isLeft else "    "), False)

    print(prefix + ("└── " if isLeft else "┌── ") + str(node.val))

    if node.left:
        prettyPrintTree(node.left, prefix + ("    " if isLeft else "│   "), True)


def treeNodeToString(root):
    if not root:
        return "[]"
    output = ""
    queue = [root]
    current = 0
    while current != len(queue):
        node = queue[current]
        current = current + 1

        if not node:
            output += "null, "
            continue

        output += str(node.val) + ", "
        queue.append(node.left)
        queue.append(node.right)
    return "[" + output[:-2] + "]"


def stringToTreeNode(input):
    """
    :param input: must be string e.g: stringToTreeNode('[1,null,2]')
    :return: TreeNode
    """
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root


# ############################### 292. Nim 游戏 #################################
# # 每次你们轮流拿掉 1 - 3 块石头
# x = 4           # F
#
#
# class Solution:
#     # 通过动态规划分析可得,只要我是4的倍数就会输..
#     def canWinNim(self, n: int) -> bool:
#         # if n % 4:
#         #     return True
#         # return False
#
#         return not (n % 4 == 0)
#
#
# solve = Solution()
# print(solve.canWinNim(x))

# ############################### 295. 数据流的中位数 #################################
import heapq


# class MedianFinder:
#
#     def __init__(self):
#         """
#         initialize your data structure here.
#         """
#         self.length = 0
#         self.max_heap = []
#         self.min_heap = []
#
#     def addNum(self, num: int) -> None:
#         self.length += 1
#         # 因为 Python 中的堆默认是小顶堆，所以用于比较的元素需是相反数才能模拟出大顶堆的效果
#         heapq.heappush(self.max_heap, -num)
#         max_heap_top = heapq.heappop(self.max_heap)
#         heapq.heappush(self.min_heap, -max_heap_top)
#         if self.length & 1:
#             min_heap_top = heapq.heappop(self.min_heap)
#             heapq.heappush(self.max_heap, -min_heap_top)
#
#     def findMedian(self) -> float:
#         if self.length & 1:
#             return -self.max_heap[0]
#         else:
#             # 如果元素个数是偶数，数据流的中位数就是各自堆顶元素的平均值
#             return (self.min_heap[0] - self.max_heap[0]) / 2


class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        if len(self.max_heap) == len(self.min_heap):
            heapq.heappush(self.min_heap, -heapq.heappushpop(self.max_heap, -num))
        else:
            heapq.heappush(self.max_heap, -heapq.heappushpop(self.min_heap, num))

    def findMedian(self) -> float:
        if len(self.max_heap) == len(self.min_heap):
            return (self.min_heap[0] - self.max_heap[0]) / 2
        else:
            return self.min_heap[0]


# Your MedianFinder object will be instantiated and called as such:
obj = MedianFinder()
obj.addNum(-1)
obj.addNum(-2)
print(obj.findMedian())     # 1.5
obj.addNum(-3)
print(obj.findMedian())     # 2
obj.addNum(-4)
print(obj.findMedian())     # 2.5
obj.addNum(-5)
print(obj.findMedian())     # 3

# ############################### 297. 二叉树的序列化与反序列化 #################################
# class Codec:
#
#     def serialize(self, root):
#         """Encodes a tree to a single string.
#
#         :type root: TreeNode
#         :rtype: str
#         """
#         if not root:
#             return '[]'
#         res = []
#         queue = [root]
#         while set(queue) != {None}:
#             tmp = queue.pop(0)
#             if tmp:
#                 res.append(str(tmp.val))
#                 queue.append(tmp.left)
#                 queue.append(tmp.right)
#             else:
#                 res.append('null')
#
#         return '[' + ','.join(res) + ']'
#
#         # 这是leetcode自带的转化函数.用列表的join函数太慢了..
#         # if not root:
#         #     return "[]"
#         # output = ""
#         # queue = [root]
#         # current = 0
#         # while current != len(queue):
#         #     node = queue[current]
#         #     current = current + 1
#         #
#         #     if not node:
#         #         output += "null, "
#         #         continue
#         #
#         #     output += str(node.val) + ", "
#         #     queue.append(node.left)
#         #     queue.append(node.right)
#         # return "[" + output[:-2] + "]"
#
#
#     def deserialize(self, data):
#         """Decodes your encoded data to tree.
#
#         :type data: str
#         :rtype: TreeNode
#         """
#         lis = data[1:-1].split(',')
#         if not lis[0]:
#             return None
#         root = TreeNode(int(lis.pop(0)))
#         queue = [root]
#         flag = 0
#         while lis:
#             tmp = lis.pop(0)
#             if tmp != 'null':
#                 node = TreeNode(int(tmp))
#                 if not flag:
#                     queue[0].left = node
#                     flag = 1
#                 else:
#                     queue[0].right = node
#                     flag = 0
#                     queue.pop(0)
#                 queue.append(node)
#             else:
#                 if not flag:
#                     flag = 1
#                 else:
#                     flag = 0
#                     queue.pop(0)
#         return root
#
#
# # Your Codec object will be instantiated and called as such:
# root = stringToTreeNode('[1,2,3,4,null,null,null,5,6]')
# root = stringToTreeNode('[5,2,3,null,null,2,4,3,1]')
# codec = Codec()
# print(codec.serialize(root))
# prettyPrintTree(codec.deserialize(codec.serialize(root)))

# ############################### 299. 猜数字游戏 #################################
# secret = "1807"
# guess = "7810"          # "1A3B"
#
# secret = "1123"
# guess = "0111"          # "1A1B"
#
# # secret = "1122"
# # guess = "0001"          # "0A1B"
#
#
# import collections
#
#
# class Solution:
#     def getHint(self, secret: str, guess: str) -> str:
#         cowA, cowB = collections.defaultdict(int),  collections.defaultdict(int)
#         bull = cow = 0
#         for a,b in zip(secret, guess):
#             if a == b:
#                 bull += 1
#                 continue
#             cowA[a] += 1
#             cowB[b] += 1
#         for k in cowA:
#             cow += min(cowA[k], cowB[k])
#         return "%dA%dB" % (bull, cow)
#
#
#         dic = {}
#         count_A = 0
#         count_B = 0
#
#         for num in secret:
#             if num not in dic:
#                 dic[num] = 1
#             else:
#                 dic[num] += 1
#
#         for i in range(len(guess)):
#             if guess[i] in dic:
#                 if guess[i] == secret[i]:
#                     count_A += 1
#                 else:
#                     count_B += 1
#                 if dic[guess[i]] < 1:
#                     count_B -= 1
#                 else:
#                     dic[guess[i]] -= 1
#         return str(count_A) + 'A' + str(count_B) + 'B'
#
#
# solve = Solution()
# print(solve.getHint(secret, guess))

# ############################### 300. 最长上升子序列 #################################
# # 你能将算法的时间复杂度降低到 O(n log n) 吗?
# x = [10,9,2,5,3,7,101,18]           # 4 ([2,3,7,101])
# x = [10,9,2,4,7,3,5,6]              # 4 ([2,3,5,6])
# # x = [1,3,6,7,9,4,10,5,6]            # 6
#
#
# class Solution:
#     def lengthOfLIS(self, nums):
#         # # O(n^2)
#         # if not nums:
#         #     return 0
#         # length = len(nums)
#         # dp = [1] * length
#         # for i, num in enumerate(nums):
#         #     for j in range(i):
#         #         if num > nums[j]:
#         #             dp[i] = max(dp[i], dp[j] + 1)
#         # print(dp)
#         # return max(dp)
#
#         # O(NlogN)二分查找重点在于保证单调区间
#         tails, res = [0] * len(nums), 0
#         for num in nums:
#             i, j = 0, res
#             while i < j:
#                 m = (i + j) // 2
#                 if tails[m] < num:
#                     i = m + 1 # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
#                 else:
#                     j = m
#             tails[i] = num
#             if j == res:
#                 res += 1
#         return res
#
#
# solve = Solution()
# print(solve.lengthOfLIS(x))
