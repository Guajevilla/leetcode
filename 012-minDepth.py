import copy
import time
import json
# ############################### 111. 二叉树的最小深度 ###############################


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


x = stringToTreeNode('[3,9,20,null,null,15,7]')
x = stringToTreeNode('[1,2]')

#
# class Solution(object):
#     def minDepth(self, root):
#         """
#         :type root: TreeNode
#         :rtype: int
#         """
#         # 递归
#         if not root:
#             return 0
#         else:
#             stack, min_depth = [(1, root), ], float('inf')
#
#         while stack:
#             depth, root = stack.pop()
#             children = [root.left, root.right]
#             if not any(children):
#                 min_depth = min(depth, min_depth)
#             for c in children:
#                 if c:
#                     stack.append((depth + 1, c))
#
#         return min_depth
#
#
#         # 迭代
#         # if not root:
#         #     return 0
#         #
#         # children = [root.left, root.right]
#         # # if we're at leaf node
#         # if not any(children):
#         #     return 1
#         #
#         # min_depth = float('inf')
#         # for c in children:
#         #     if c:
#         #         min_depth = min(self.minDepth(c), min_depth)
#         # return min_depth + 1
#
#
#         # length = [float('inf')]
#         # if not root:
#         #     return 0
#         #
#         # def find_depth(root, depth):
#         #     if root.left:
#         #         find_depth(root.left, depth + 1)
#         #     if root.right:
#         #         find_depth(root.right, depth + 1)
#         #     if not (root.left or root.right):
#         #         if depth < length[0]:
#         #             length[0] = depth
#         #
#         # find_depth(root, 1)
#         # return length[0]
#
#
# solve = Solution()
# print(solve.minDepth(x))

# ############################### 112. 路径总和 ###############################
# root = stringToTreeNode('[5,4,8,11,null,13,4,7,2,null,null,null,1]')
# sum = 22        # T
# root = stringToTreeNode('[1,2]')
# sum = 1        # F
#
#
# class Solution(object):
#     def hasPathSum(self, root, sum):
#         """
#         :type root: TreeNode
#         :type sum: int
#         :rtype: bool
#         """
#         # res = []
#         #
#         # def sub_root(root, sub_sum):
#         #     if not root:
#         #         return
#         #     sub_sum += root.val
#         #     if not (root.left or root.right):
#         #         res.append(sub_sum)
#         #
#         #     sub_root(root.left, sub_sum)
#         #     sub_root(root.right, sub_sum)
#         #
#         # if not root:
#         #     return False
#         # sub_root(root, 0)
#         # return sum in res
#
#
#         if not root:
#             return False
#
#         sum -= root.val
#         if not root.left and not root.right:  # if reach a leaf
#             return sum == 0
#         return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)
#
#
# solve = Solution()
# print(solve.hasPathSum(root, sum))

# ############################### 113. 路径总和 II ###############################
# root = stringToTreeNode('[5,4,8,11,null,13,4,7,2,null,null,5,1]')
# sums = 22
# # [
# #    [5,4,11,2],
# #    [5,8,4,5]
# # ]
# # root = stringToTreeNode('[1,2]')
# # sum = 1
#
#
# class Solution(object):
#     def pathSum(self, root, sums):
#         """
#         :type root: TreeNode
#         :type sums: int
#         :rtype: List[List[int]]
#         """
#         # 这个地方有个大Bug。。原题里的变量占用了sum关键字。。
#         # res = []
#         #
#         # def sub_root(root, route):
#         #     if not root:
#         #         return
#         #     if not (root.left or root.right):
#         #         if sum(route) + root.val == sums:
#         #             res.append(route + [root.val])
#         #         else:
#         #             return
#         #
#         #     sub_root(root.left, route + [root.val])
#         #     sub_root(root.right, route + [root.val])
#         #
#         # if not root:
#         #     return []
#         # sub_root(root, [])
#         # return res
#
#         res = []
#         if not root: return []
#
#         def helper(root, sums, tmp):
#             if not root:
#                 return
#             if not root.left and not root.right and sums - root.val == 0:
#                 tmp += [root.val]
#                 res.append(tmp)
#                 return
#             helper(root.left, sums - root.val, tmp + [root.val])
#             helper(root.right, sums - root.val, tmp + [root.val])
#
#         helper(root, sums, [])
#         return res
#
#
# solve = Solution()
# print(solve.pathSum(root, sums))

# ############################### 114. 二叉树展开为链表 ###############################
root = stringToTreeNode('[1,2,5,3,4,null,6]')
# prettyPrintTree(root)           #　[1,null,2,null,3,null,4,null,5,null,6]
# root = stringToTreeNode('[1,null,2,null,3,null,4,null,5,null,6]')
# prettyPrintTree(root)


class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
# #         先先序遍历，把值存起来再建立,感觉不应该这么做
#         res = []
#
#         def pre_order(root):
#             if not root:
#                 return
#             res.append(root.val)
#             pre_order(root.left)
#             pre_order(root.right)
#
#         if not root:
#             return root
#         pre_order(root)
#         root.left = None
#         res.pop(0)
#         while res:
#             root.right = TreeNode(res.pop(0))
#             root = root.right

        # 由底向上，递归, 类似后序遍历
        def helper(root, pre):
            if not root: return pre
            # 记录遍历时候,该节点的前一个节点
            pre = helper(root.right, pre)
            pre = helper(root.left, pre)
            # 拼接
            root.right = pre
            root.left = None
            return pre

        helper(root, None)

        # 自上向下，不断将右子树放在左子树的最右叶子节点上，再把左子树放在右子树处
        while root:
            if root.left:
                most_right = root.left
                while most_right.right:
                    most_right = most_right.right
                most_right.right = root.right
                root.right = root.left
                root.left = None
            root = root.right
        return


solve = Solution()
solve.flatten(root)
print(treeNodeToString(root))

# ############################### 115. 不同的子序列 ###############################
# S = "rabbbit"
# T = "rabbit"        # 3
#
# # S = "babgbag"
# # T = "bag"           # 5
#
#
# class Solution(object):
#     def numDistinct(self, s, t):
#         """
#         :type s: str
#         :type t: str
#         :rtype: int
#         """
#         dp = [[1] * (len(s) + 1)]
#         for _ in range(len(t)):
#             dp.append([0] * (len(s) + 1))
#
#         for i in range(1, len(t) + 1):
#             for j in range(i, len(s) + 1):
#                 if t[i-1] == s[j-1]:
#                     dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1]
#                 else:
#                     dp[i][j] = dp[i][j - 1]
#
#         return dp[-1][-1]
#
#
# solve = Solution()
# print(solve.numDistinct(S, T))

# ############################### 116. 填充每个节点的下一个右侧节点指针 ###############################
#
#
# # Definition for a Node.
# class Node(object):
#     def __init__(self, val, left, right, next):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next
#
#
# class Solution(object):
#     def connect(self, root):
#         """
#         :type root: Node
#         :rtype: Node
#         """
#         if not root:
#             return
#         if root.left:
#             root.left.next = root.right
#             if root.next:
#                 root.right.next = root.next.left
#         self.connect(root.left)
#         self.connect(root.right)
#         return root
#
#
#         pre = root
#         while pre:
#             cur = pre
#             while cur:
#                 if cur.left: cur.left.next = cur.right
#                 if cur.right and cur.next: cur.right.next = cur.next.left
#                 cur = cur.next
#             pre = pre.left
#         return root
#
#
#         # # 使用的不是常数空间复杂度
#         # res = []
#         #
#         # def helper(root, depth):
#         #     if not root: return
#         #     if len(res) == depth:
#         #         res.append([])
#         #     res[depth].append(root)
#         #     helper(root.left, depth + 1)
#         #     helper(root.right, depth + 1)
#         #
#         # helper(root, 0)
#         # for lis in res:
#         #     for ind, ele in enumerate(lis):
#         #         if ind == len(lis) - 1:
#         #             ele.next = None
#         #         else:
#         #             ele.next = lis[ind + 1]
#         # return root
#
#
# solve = Solution()
# treeNodeToString(solve.connect(root))

# ############################### 117. 填充每个节点的下一个右侧节点指针 II ###############################
# 与上一题差别在于不是“完美二叉树”,这题用上一题的非常数空间复杂度解法，不用改任何地方就能过，但还是空间复杂度不符合要求


# class Solution(object):
#     def connect(self, root):
#         """
#         :type root: Node
#         :rtype: Node
#         """
#         cur = root
#         head = None
#         tail = None
#         while cur:
#             while cur:
#                 if cur.left:
#                     if not head:
#                         head = cur.left
#                         tail = cur.left
#                     else:
#                         tail.next = cur.left
#                         tail = tail.next
#                 if cur.right:
#                     if not head:
#                         head = cur.right
#                         tail = cur.right
#                     else:
#                         tail.next = cur.right
#                         tail = tail.next
#                 cur = cur.next
#             cur = head
#             head = None
#             tail = None
#         return root


# ############################### 118. 杨辉三角 ###############################
# x = 5
# # [
# #      [1],
# #     [1,1],
# #    [1,2,1],
# #   [1,3,3,1],
# #  [1,4,6,4,1]
# # ]
#
#
# class Solution(object):
#     def generate(self, numRows):
#         """
#         :type numRows: int
#         :rtype: List[List[int]]
#         """
#         # if numRows < 1:
#         #     return []
#         # res = [[1]]
#         # while numRows > 1:
#         #     numRows -= 1
#         #     tmp1 = res[-1] + [0]
#         #     tmp2 = [0] + res[-1]
#         #     for i in range(len(tmp1)):
#         #         tmp1[i] += tmp2[i]
#         #     res.append(tmp1)
#         # return res
#
#         # 模拟过程
#         res = []
#         tmp = []
#         for _ in range(numRows):
#             tmp.insert(0, 1)
#             for i in range(1, len(tmp) - 1):
#                 tmp[i] = tmp[i] + tmp[i + 1]
#             res.append(tmp[:])
#         return res
#
#
# solve = Solution()
# print(solve.generate(x))

# ############################### 119. 杨辉三角 II ###############################
# x = 1       # [1,3,3,1]
#
#
# class Solution(object):
#     def getRow(self, rowIndex):
#         """
#         :type rowIndex: int
#         :rtype: List[int]
#         """
#         # res = [1]
#         # while rowIndex > 0:
#         #     rowIndex -= 1
#         #     tmp = res + [0]
#         #     res = [0] + res
#         #     for i in range(len(res)):
#         #         res[i] += tmp[i]
#         # return res
#
#         # 模拟过程
#         tmp = []
#         for _ in range(rowIndex + 1):
#             tmp.insert(0, 1)
#             for i in range(1, len(tmp) - 1):
#                 tmp[i] = tmp[i] + tmp[i+1]
#         return tmp
#
#
# solve = Solution()
# print(solve.getRow(x))

# ############################### 120. 三角形最小路径和 ###############################
# 只使用 O(n) 的额外空间（n 为三角形的总行数）
x = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]               # 11

x = [
     [2],
    [3,4],
   [6,5,1],
  [4,1,8,3]
]               # 10

# x = []               # 10


class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        # # 为了节省空间改变了原数据
        # if not triangle or not triangle[0]:
        #     return 0
        # for i in range(1, len(triangle)):
        #     for j in range(len(triangle[i])):
        #         if j == 0:
        #             triangle[i][j] += triangle[i - 1][j]
        #         elif j == len(triangle[i]) - 1:
        #             triangle[i][j] += triangle[i - 1][-1]
        #         else:
        #             triangle[i][j] += min(triangle[i - 1][j], triangle[i - 1][j - 1])
        #
        # return min(triangle[-1])

        # 只存一个dp = triangle[-1]从下向上比,dp的有效位数逐渐减小到1,最后返回dp[0]
        row = len(triangle)
        dp = triangle[-1]
        for i in range(row-2, -1, -1):
            for j in range(i + 1):
                dp[j] = min(dp[j], dp[j+1])+triangle[i][j]
        return dp[0]


solve = Solution()
print(solve.minimumTotal(x))
print(x)
