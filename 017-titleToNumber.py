import math
import json
# ############################### 171. Excel表列序号 ###############################
# x = 'A'         #１
# # x = 'AB'        # 28
# # x = 'ZY'        # 701
# # x = 'ZZ'        # 702
# # x = 'BZB'       # 2030
# x = ''
#
#
# class Solution:
#     def titleToNumber(self, s: str) -> int:
#         res = 0
#         for ele in s:
#             res = res * 26 + ord(ele) - ord('A') + 1
#         return res
#
#
# solve = Solution()
# print(solve.titleToNumber(x))

# ############################### 172. 阶乘后的零 ###############################
# x = 3       # 0
# # x = 5       # 1
# x = 25       # 6
# x = 75       # 18
# x = 126       # 31
#
#
# class Solution:
#     def trailingZeroes(self, n: int) -> int:
#         # 每有一个5,尾数就多一个0
#         # 5的 n 次方有 n个5
#         res = 0
#         fact = 5
#         while n // fact:
#             res += n // fact
#             fact *= 5
#         return res
#
#
#         p = 0
#         while n >= 5:
#             n = n // 5
#             p += n
#         return p
#
#
# solve = Solution()
# print(solve.trailingZeroes(x))

# ############################### 173. 二叉搜索树迭代器 ###############################


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
# 173. 二叉搜索树迭代器
# next() 和 hasNext() 操作的时间复杂度是 O(1)，并使用 O(h) 内存，其中 h 是树的高度。
# 你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 中至少存在一个下一个最小的数。
root = stringToTreeNode('[7,3,15,null,null,9,20]')
prettyPrintTree(root)


class BSTIterator:
    #  要求 O(h) 内存,所以不能中序遍历,可以采用莫里斯遍历
    def __init__(self, root: TreeNode):
        self.root = root
        self.stack = [root]
        # tmp = [root.val]
        while self.stack[-1].left:
            # tmp.append(self.stack[-1].left.val)
            self.stack.append(self.stack[-1].left)
        # print(tmp)

    def next(self) -> int:
        """
        @return the next smallest number
        """
        if not self.stack[-1].right:
            tmp = self.stack.pop()
            if self.stack:
                self.stack[-1].left = None
        elif not self.stack[-1].left:
            tmp = self.stack.pop()
            self.stack.append(tmp.right)
        else:
            # 这里的问题在于其实是需要遍历左子树的,还涉及到后面的右子树
            tmp = self.stack[-1].left
            self.stack[-1].left = None

        return tmp.val

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return len(self.stack) > 0


obj = BSTIterator(root)
print(obj.next())       # 3
print(obj.hasNext())    # T
print(obj.next())       # 7
print(obj.hasNext())    # T
print(obj.next())       # 9
print(obj.hasNext())    # T
print(obj.next())       # 15
print(obj.hasNext())    # T
print(obj.next())       # 20
print(obj.hasNext())    # F

# ############################### 174. 地下城游戏 ###############################
# x = [[-2,-3, 3],
#      [-5,-10,1],
#      [10,30,-5]]        # 7
#
# # x = [[2,-3, 3, 4],
# #      [-5,-10,1,-3],
# #      [10,30,-3,-5]]        # 7
#
#
# class Solution:
#     def calculateMinimumHP(self, dungeon):
#         # # 这题不一样的地方在于必须要从终点往前推
#         # dp = [[0] * len(dungeon[0]) for _ in range(len(dungeon))]
#         # dp[-1][-1] = max(1, 1 - dungeon[-1][-1])
#         # for j in range(2, len(dungeon[0]) + 1):
#         #     dp[-1][-j] = max(1, dp[-1][-j + 1] - dungeon[-1][-j])
#         #
#         # for i in range(2, len(dungeon) + 1):
#         #     dp[-i][-1] = max(1, dp[-i + 1][-1] - dungeon[-i][-1])
#         #
#         # for i in range(2, len(dungeon) + 1):
#         #     for j in range(2, len(dungeon[0]) + 1):
#         #         dp[-i][-j] = max(1, min(dp[-i + 1][-j], dp[-i][-j + 1]) - dungeon[-i][-j])
#         # print(dp)
#         # return dp[0][0]
#
#         # 只用最后一行大小也能dp
#         m, n = len(dungeon), len(dungeon[0])
#         dp = [0] * n
#         # 初始化最后一行
#         dp[-1] = max(1, 1 - dungeon[-1][-1])
#         for i in range(n - 2, -1, -1):
#             dp[i] = max(1, dp[i + 1] - dungeon[-1][i])
#         for j in range(m - 2, -1, -1):
#             dp[-1] = max(1, dp[-1] - dungeon[j][-1])
#             for k in range(n - 2, -1, -1):
#                 dp[k] = max(1, min(dp[k], dp[k + 1]) - dungeon[j][k])
#         return dp[0]
#
#
# solve = Solution()
# print(solve.calculateMinimumHP(x))

# ############################### 174. 地下城游戏 ###############################
