# ############################### 121. 买卖股票的最佳时机 ###############################
# x = [7,1,5,3,6,4]       # 5
# x = []       # 5
# # x = [7,6,4,3,1]       # 0
#
#
# class Solution(object):
#     def maxProfit(self, prices):
#         """
#         :type prices: List[int]
#         :rtype: int
#         """
#         # 用dp的思想,每一步找到最大
#         min_p, max_p = float('inf'), 0
#         for i in range(len(prices)):
#             min_p = min(min_p, prices[i])
#             max_p = max(max_p, prices[i] - min_p)
#         return max_p
#
#
#         # # 超时
#         # profit = 0
#         # for i_in, i_price in enumerate(prices[:-1]):
#         #     o_price = max(prices[i_in+1:])
#         #     profit = max(profit, (o_price - i_price))
#         # return profit
#
#
# solve = Solution()
# print(solve.maxProfit(x))

# ############################### 122. 买卖股票的最佳时机 II ###############################
# x = [7,1,5,3,6,4]       # 7
# x = [1,2,3,4,5]         # 4
# x = [7,1,2,1,7]         # 7
# # x = [7,6,4,3,1]       # 0
#
#
# class Solution(object):
#     def maxProfit(self, prices):
#         """
#         :type prices: List[int]
#         :rtype: int
#         """
#         profit = 0
#         for i in range(1, len(prices)):
#             tmp = prices[i] - prices[i - 1]
#             if tmp > 0:
#                 profit += tmp
#         return profit
#
#
# solve = Solution()
# print(solve.maxProfit(x))

# ############################### 123. 买卖股票的最佳时机 III ###############################
x = [3,3,5,0,0,3,1,4]       # 6
# x = [1,2,3,4,5]         # 4
# x = [7,6,4,3,1]       # 0
x = [1,2,4,2,5,7,2,4,9,0]   # 13
# x = [1,2]
# x = [1,4,2]


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # # 最后一个用例超时...
        # if not prices:
        #     return 0
        # profit = 0
        # min_prices = prices[0]
        # max_i = 0
        # for i, cur_price in enumerate(prices[1:]):
        #     # 第i+1天卖出的最大利润
        #     if cur_price < min_prices:
        #         min_prices = cur_price
        #         continue
        #     else:
        #         max_i = max(cur_price - min_prices, max_i)
        #     # max_i = 0
        #     # for pre in prices[:i+1]:
        #     #     max_i = max(max_i, cur_price - pre)
        #     # 后面的这几天找最大,利用121的解法
        #     min_p, max_p = float('inf'), 0
        #     for after_price in prices[i + 2:]:
        #         min_p = min(min_p, after_price)
        #         max_p = max(max_p, after_price - min_p)
        #     profit = max(max_p + max_i, profit)
        #
        # return profit


        # if not prices:
        #     return 0
        # K = 2
        # dp = [[0] * (K + 1) for i in range(len(prices))]
        # for k in range(1, K + 1):
        #     min_price = prices[0]
        #     for i in range(1, len(prices)):
        #         min_price = min(prices[i] - dp[i][k - 1], min_price)
        #         dp[i][k] = max(prices[i] - min_price, dp[i - 1][k])
        #
        # return dp[-1][-1]

        if len(prices) == 0:
            return 0
        fstBuy, fstSell, secBuy, secSell = prices[0], 0, prices[0], 0

        for i in range(len(prices)):
            fstBuy = min(fstBuy, prices[i])
            fstSell = max(fstSell, prices[i] - fstBuy)
            secBuy = min(secBuy, prices[i] - fstSell)
            secSell = max(secSell, prices[i] - secBuy)
        return secSell


solve = Solution()
print(solve.maxProfit(x))

# ############################### 124. 二叉树中的最大路径和 ###############################
# 本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
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


x = stringToListNode('[1,5,9]')


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


# x = stringToTreeNode('[-2,-1]')     # 6
# # x = stringToTreeNode('[2,-1]')     # 2
# # x = stringToTreeNode('[-2,1]')     # 3
# # x = stringToTreeNode('[3,2,1]')     # 6
# # x = stringToTreeNode('[-10,9,20,null,null,15,7]')     # 42
# # x = stringToTreeNode('[-10,45,20,null,null,15,7]')     # 70
# # x = stringToTreeNode('[-2,null,-3]')     # -2
# # x = stringToTreeNode('[1,-2,-3,1,3,-2,null,-1]')     # 3
#
#
# class Solution(object):
#     def maxPathSum(self, root):
#         """
#         :type root: TreeNode
#         :rtype: int
#         """
#         # 其实思想和我写的一样,但是简洁太多了..
#         self.res = float("-inf")
#
#         def helper(root):
#             if not root: return 0
#             # 右边最大值
#             left = helper(root.left)
#             # 左边最大值
#             right = helper(root.right)
#             # 和全局变量比较
#             self.res = max(left + right + root.val, self.res)
#             # >0 说明都能使路径变大
#             return max(0, max(left,  right) + root.val)
#         helper(root)
#         return self.res
#
#         # # 利用后序遍历的顺序是自底向上,分四种情况考虑:没有左右节点,只有左\右,都有
#         # # 每个节点存以该节点为根的子树的包含根节点的最大路径
#         # # 时间复杂度较高,其中几个max的处理应该可以通过比较优化
#         # res = [float('-inf')]
#         #
#         # def sub_path(root):
#         #     if not root:
#         #         return
#         #     if not (root.left or root.right):
#         #         res[0] = max(root.val, res[0])
#         #     else:
#         #         if root.left and root.right:
#         #             sub_max = max(root.left.val, root.right.val)
#         #             res[0] = max(res[0], sub_max, root.val, sub_max + root.val, root.left.val + root.right.val + root.val)
#         #             root.val = max(root.val + sub_max, root.val)
#         #         elif root.left:
#         #             root.val = max(root.val, root.val + root.left.val)
#         #             res[0] = max(root.val, res[0], root.left.val)
#         #         else:
#         #             root.val = max(root.val, root.val + root.right.val)
#         #             res[0] = max(root.val, res[0], root.right.val)
#         #
#         # def post_order(root):
#         #     if not root:
#         #         return
#         #     post_order(root.left)
#         #     post_order(root.right)
#         #     sub_path(root)
#         #
#         # post_order(root)
#         # prettyPrintTree(root)
#         # return res[0]
#
#         # # 利用中序遍历,不行，这样包括了重复回路
#         # def find_sub(root):
#         #     if not root:
#         #         return
#         #     find_sub(root.left)
#         #     res.append(root.val)
#         #     find_sub(root.right)
#         #
#         # find_sub(root)
#         # print(res)
#         # for i in range(1, len(res)):
#         #     res[i] = max(res[i - 1] + res[i], res[i])
#         # print(res)
#         # return max(res)
#
#
# solve = Solution()
# print(solve.maxPathSum(x))

# ############################### 125. 验证回文串 ###############################
# x = "A man, a plan, a canal: Panama"    # T
# # x = "race a car"                        # F
# # x = "11"
# # x = ".,"
#
#
# class Solution(object):
#     def isPalindrome(self, s):
#         """
#         :type s: str
#         :rtype: bool
#         """
#         # 正则表达式
#         import re
#         tmp = re.sub(r"[^A-Za-z0-9]","", s).lower()
#         return tmp == tmp[::-1]
#
#
#         # s = s.lower()
#         # if len(s) == 0 or len(s) == 1:
#         #     return True
#         # head = 0
#         # tail = len(s) - 1
#         # while head <= tail:
#         #     while head <= tail and not (s[head].isdigit() or s[head].isalpha()):    # .isalnum()
#         #         head += 1
#         #     if head >= tail:
#         #         return True
#         #     while head <= tail and not (s[tail].isdigit() or s[tail].isalpha()):
#         #         tail -= 1
#         #     if head >= tail:
#         #         return True
#         #     if s[head] != s[tail]:
#         #         return False
#         #     else:
#         #         head += 1
#         #         tail -= 1
#         # return True
#
#
# solve = Solution()
# print(solve.isPalindrome(x))

# ############################### 126. 单词接龙 II ###############################
# 每次转换只能改变一个字母。
# 所有单词具有相同的长度。
# 所有单词只由小写字母组成。
# 字典中不存在重复的单词。
# 你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
# [
#   ["hit","hot","dot","dog","cog"],
#   ["hit","hot","lot","log","cog"]
# ]

# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log"]
# # []


class Solution(object):
    def find_similarity(self, word1, word2):
        cnt = 0
        for i in range(len(word1)):
            if word1[i] == word2[i]:
                cnt += 1
        return cnt

    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        newList = [[beginWord], [endWord]]
        n = len(beginWord)
        length = n - self.find_similarity(beginWord, endWord)
        begin_list = [[beginWord]]
        begin_list.extend([] for _ in range(length))
        end_list = [[] for _ in range(length)]
        end_list.insert(0, [beginWord])
        for word in wordList:
            begin_similarity = self.find_similarity(word, beginWord)
            end_similarity = self.find_similarity(word, endWord)
            begin_list[n - begin_similarity].append(word)
            end_list[end_similarity].append(word)
            # if begin_similarity + end_similarity == n:
            #     return
        print(begin_list)
        print(end_list)


solve = Solution()
print(solve.findLadders(beginWord, endWord, wordList))

# ############################### 127. 单词接龙 ###############################
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
# 5

beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
# 0


class Solution(object):
    def find_one_difference(self):
        return

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """


# solve = Solution()
# print(solve.ladderLength(beginWord, endWord, wordList))

# ############################### 128. 最长连续序列 ###############################
nums = [100, 4, 200, 1, 3, 2]       # 4


class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """


# solve = Solution()
# print(solve.longestConsecutive(nums))

# ############################### 129. 求根到叶子节点数字之和 ###############################
# x = stringToTreeNode('[1,2,3]')      # 25
# x = stringToTreeNode('[4,9,0,5,1]')  # 1026
# x = stringToTreeNode('[1]')      # 25
#
#
# class Solution(object):
#     def sumNumbers(self, root):
#         """
#         :type root: TreeNode
#         :rtype: int
#         """
#         res = [0]
#
#         def pre_order(root, s):
#             if not root:
#                 return
#             s = 10*s + root.val
#             if not (root.left or root.right):
#                 res[0] += s
#
#             pre_order(root.left, s)
#             pre_order(root.right, s)
#
#         if not root:
#             return 0
#         pre_order(root, 0)
#         print(res)
#         return res[0]
#
#
# solve = Solution()
# print(solve.sumNumbers(x))

# ############################### 130. 被围绕的区域 ###############################
board = [['X','X','X','X'],
         ['X','O','O','X'],
         ['X','X','O','X'],
         ['X','O','X','X']]
# [['X X X X'],
#  ['X X X X'],
#  ['X X X X'],
#  ['X O X X']]


class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        dp = [[1] * len(board[0]) for _ in range(len(board))]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if i == 0 or j == 0 or i == len(board) - 1 or j == len(board[0]) - 1:
                    dp[i][j] = board[i][j] == 'X'
                else:
                    if board[i][j] == 'X':
                        dp[i][j] = 1
                    else:
                        if dp[i - 1][j] and dp[i][j - 1]:
                            dp[i][j] = 2
                        else:
                            dp[i][j] = 0


# solve = Solution()
# solve.solve(board)
# print(board)
