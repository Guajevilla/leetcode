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
x = [1,2,3,4,5]         # 4
x = [7,6,4,3,1]       # 0
x = [1,2,4,2,5,7,2,4,9,0]   # 13


class Solution(object):
    def insert_profit(self, profit, tmp_profit):
        if not profit:
            profit.append(tmp_profit)
        elif len(profit) == 1:
            if tmp_profit > profit[0]:
                profit.append(tmp_profit)
            else:
                profit.insert(0, tmp_profit)
        else:
            if tmp_profit > profit[0]:
                profit.pop(0)
                if tmp_profit > profit[0]:
                    profit.append(tmp_profit)
                else:
                    profit.insert(0, tmp_profit)

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = []
        tmp_profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0:
                tmp_profit += tmp
            else:
                self.insert_profit(profit, tmp_profit)
                tmp_profit = 0

        if tmp_profit > 0:
            self.insert_profit(profit, tmp_profit)

        print(profit)
        return sum(profit)

    # def maxProfit(self, prices):
    #     if not prices: return 0
    #     n = len(prices)
    #     dp = [[0] * n for _ in range(3)]
    #     for k in range(1, 3):
    #         pre_max = -prices[0]
    #         for i in range(1, n):
    #             pre_max = max(pre_max, dp[k - 1][i - 1] - prices[i])
    #             dp[k][i] = max(dp[k][i - 1], prices[i] + pre_max)
    #     return dp[-1][-1]



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
x = "A man, a plan, a canal: Panama"    # T
x = "race a car"                        # F
x = "1"


class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 0 or 1:# len(s) == 1:
            return True


solve = Solution()
print(solve.isPalindrome(x))
