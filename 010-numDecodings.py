import copy
import functools
import time
# ############################### 91. 解码方法 ###############################
# nums = "12"         # 2
# nums = "226"        # 3
# nums = "01"         # 0
# nums = "30405060"   # 0
# nums = "272829"     # 1
# nums = "31324145"   # 8
# nums = "31224145"   # 10
# nums = "010"        # 0
# nums = "3122414"      # 10
# nums = "101"      # 1
# nums = "110"      # 1
# nums = "2103"      # 1
#
#
# class Solution(object):
#     def numDecodings(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         if s[0] == '0':
#             return 0
#         dp = [1, 1] + [0] * (len(s) - 1)
#         for i in range(1, len(s)):
#             if s[i] == '0':
#                 if s[i-1] != '1' and s[i-1] != '2':
#                     return 0
#                 else:
#                     dp[i+1] = dp[i-1]
#                     continue
#
#             if 10 < int(s[i-1:i+1]) <= 26:
#                 dp[i+1] = dp[i] + dp[i-1]
#             else:
#                 dp[i+1] = dp[i]
#
#         return dp[-1]
#
#
# # class Solution:
# #     def numDecodings(self, s: str) -> int:
# #         dp = [0] * len(s)
# #         # 考虑第一个字母
# #         if s[0] == "0":
# #             return 0
# #         else:
# #             dp[0] = 1
# #         if len(s) == 1: return dp[-1]
# #         # 考虑第二个字母
# #         if s[1] != "0":
# #             dp[1] += 1
# #         if 10 <= int(s[:2]) <= 26:
# #             dp[1] += 1
# #         for i in range(2, len(s)):
# #             # 当出现连续两个0
# #             if s[i - 1] + s[i] == "00": return 0
# #             # 考虑单个字母
# #             if s[i] != "0":
# #                 dp[i] += dp[i - 1]
# #             # 考虑两个字母
# #             if 10 <= int(s[i - 1] + s[i]) <= 26:
# #                 dp[i] += dp[i - 2]
# #         return dp[-1]
#
#
# solve = Solution()
# print(solve.numDecodings(nums))

# # ############################### 92. 反转链表 II ###############################
# # 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
# # 1 ≤ m ≤ n ≤ 链表长度。
#
#
# # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
#
# def printList(l: ListNode):
#     while l:
#         print("%d, " %(l.val), end = '')
#         l = l.next
#     print('')
#
#
# x = ListNode(1)
# x.next = ListNode(2)
# x.next.next = ListNode(3)
# x.next.next.next = ListNode(4)
# x.next.next.next.next = ListNode(5)
# m = 2
# n = 4                   # 1->4->3->2->5
#
# x = ListNode(1)
# x.next = ListNode(2)
# # x.next.next = ListNode(3)
# # x.next.next.next = ListNode(4)
# # x.next.next.next.next = ListNode(5)
# m = 1
# n = 2                   # 1->4->3->2->5
#
#
# class Solution(object):
#     def reverseBetween(self, head, m, n):
#         """
#         :type head: ListNode
#         :type m: int
#         :type n: int
#         :rtype: ListNode
#         """
#         i = 0
#         dummy = ListNode(0)
#         dummy.next = head
#         stack = []
#         flag = 0
#         head = dummy
#         while head:
#             i += 1
#             if i == m:
#                 pre = head
#                 flag = 1
#             elif flag == 1:
#                 if i <= n + 1:
#                     stack.append(head.val)
#             if i == n + 1:
#                 while stack:
#                     pre.next = ListNode(stack.pop())
#                     pre = pre.next
#                 pre.next = head.next
#                 break
#             head = head.next
#
#         return dummy.next
#
#         # 用三个指针,进行插入操作
#         # 例如:
#         # 1->2->3->4->5->NULL, m = 2, n = 4
#         # 将节点3插入节点1和节点2之间
#         # 变成: 1->3->2->4->5->NULL
#         # 再将节点4插入节点1和节点3之间
#         # 变成:1->4->3->2->5->NULL
#         # def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
#         #     dummy = ListNode(-1)
#         #     dummy.next = head
#         #     pre = dummy
#         #     # 找到翻转链表部分的前一个节点, 1->2->3->4->5->NULL, m = 2, n = 4 指的是 节点值为1
#         #     for _ in range(m - 1):
#         #         pre = pre.next
#         #     # 用 pre, start, tail三指针实现插入操作
#         #     # tail 是插入pre,与pre.next的节点
#         #     start = pre.next
#         #     tail = start.next
#         #     for _ in range(n - m):
#         #         start.next = tail.next
#         #         tail.next = pre.next
#         #         pre.next = tail
#         #         tail = start.next
#         #     return dummy.next
#
#
# solve = Solution()
# printList(solve.reverseBetween(x, m, n))

# ############################### 93. 复原IP地址 ###############################
# x = "25525511135"       # ["255.255.11.135", "255.255.111.35"]
# x = "010010"            # ["0.10.0.10","0.100.1.0"]
# # x = "101023"            # ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
# # x = ""
#
#
# class Solution(object):
#     def restoreIpAddresses(self, s):
#         """
#         :type s: str
#         :rtype: List[str]
#         """
#         # 回溯
#         res = []
#         n = len(s)
#
#         def backtrack(i, tmp, flag):
#             if i == n and flag == 0:
#                 res.append(tmp[:-1])
#                 return
#             if flag < 0:
#                 return
#             for j in range(i, i + 3):
#                 if j < n:
#                     if i == j and s[j] == "0":
#                         backtrack(j + 1, tmp + s[j] + ".", flag - 1)
#                         break
#                     if 0 < int(s[i:j + 1]) <= 255:
#                         backtrack(j + 1, tmp + s[i:j + 1] + ".", flag - 1)
#
#         backtrack(0, "", 4)
#         return res
#
#         # # 暴力解法..
#         # res = []
#         # for i in range(3):
#         #     for j in range(i+1, i+4):
#         #         for k in range(j+1, j+4):
#         #             if k < len(s) - 1:
#         #                 tmp1 = s[:i + 1]
#         #                 tmp2 = s[i + 1:j + 1]
#         #                 tmp3 = s[j + 1:k + 1]
#         #                 tmp4 = s[k + 1:]
#         #                 if max(int(tmp1), int(tmp2), int(tmp3), int(tmp4)) <= 255:
#         #                     if tmp1[0] == '0' and len(tmp1) > 1:
#         #                         continue
#         #                     elif tmp2[0] == '0' and len(tmp2) > 1:
#         #                         continue
#         #                     elif tmp3[0] == '0' and len(tmp3) > 1:
#         #                         continue
#         #                     elif tmp4[0] == '0' and len(tmp4) > 1:
#         #                         continue
#         #                     res.append(".".join([tmp1, tmp2, tmp3, tmp4]))
#         # return res
#
#         # 只能拼出前面的位数尽可能多的组合
#         # res = []
#         # last = 0
#         # for i in range(4):
#         #     j = len(s) - last - 3*(3-i)
#         #     if s[last] == '0':
#         #         res.append('0')
#         #         last += 1
#         #         continue
#         #     if j <= 0:
#         #         tmp = ''
#         #         j = last
#         #     else:
#         #         tmp = s[last:last+j]
#         #         j += last
#         #     while 3 - i <= len(s) - 1 - j:# <= (3 - i) * 3:
#         #         if s[j] == '0':
#         #             if 0 < len(tmp) < 3:
#         #                 tmp += s[j]
#         #                 j += 1
#         #                 last = j
#         #                 continue
#         #             elif len(tmp) == 0:
#         #                 tmp = '0'
#         #             else:
#         #                 break
#         #
#         #         if 0 <= int(tmp + s[j]) <= 255:
#         #             tmp += s[j]
#         #         else:
#         #             break
#         #         j += 1
#         #         last = j
#         #     res.append(tmp)
#         # return res
#
#
# solve = Solution()
# print(solve.restoreIpAddresses(x))

# ############################### 94. 二叉树的中序遍历 ###############################
#
#
# # # Definition for singly-linked list.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
#
# def create_tree(lis):
#     root = TreeNode(lis[0])
#     stack = [root]
#     for i, ele in enumerate(lis):
#         if i == 0:
#             continue
#         if i % 2:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].left = tmp
#             stack.append(tmp)
#         else:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].right = tmp
#             stack.append(tmp)
#     return root
#
#
# def print_tree(root: TreeNode):
#     res = []
#     queue = [root]
#     while queue:
#         tmp = queue.pop(0)
#         res.append(tmp.val)
#         if tmp.left:
#             queue.append(tmp.left)
#         if tmp.right:
#             queue.append(tmp.right)
#
#     print(res)
#
#
# # # 前序遍历
# # def print_tree(root: TreeNode):
# #     res = []
# #
# #     def helper(root: TreeNode):
# #         if root:
# #             res.append(root.val)
# #             helper(root.left)
# #             helper(root.right)
# #
# #     helper(root)
# #     print(res)
#
#
# class Solution(object):
#     def inorderTraversal(self, root):
#         """
#         :type root: TreeNode
#         :rtype: List[int]
#         """
#         # 迭代 因为需要不断回溯,所以需要压栈记住之前的节点,并记住之前已经进过的节点
#         res = []
#         stack = [root]
#         rem = []
#         while stack:
#             p = stack.pop()
#             if p.left and p not in rem:
#                 stack.append(p)
#                 rem.append(p)
#                 stack.append(p.left)
#             else:
#                 res.append(p.val)
#                 if p.right:
#                     stack.append(p.right)
#         return res
#
#         # # 别人的迭代
#         # res = []
#         # stack = []
#         # # 用p当做指针
#         # p = root
#         # while p or stack:
#         #     # 把左子树压入栈中
#         #     while p:
#         #         stack.append(p)
#         #         p = p.left
#         #     # 输出 栈顶元素
#         #     p = stack.pop()
#         #     res.append(p.val)
#         #     # 看右子树
#         #     p = p.right
#         # return res
#
#
#         # # 递归很好写
#         # res = []
#         #
#         # def helper(root):
#         #     if root:
#         #         helper(root.left)
#         #         res.append(root.val)
#         #         helper(root.right)
#         #
#         # helper(root)
#         # return res
#
#
# solve = Solution()
# print(solve.inorderTraversal(x))

# ############################### 95. 不同的二叉搜索树 II ###############################


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.right = None


# def create_tree(lis):
#     root = TreeNode(lis[0])
#     stack = [root]
#     for i, ele in enumerate(lis):
#         if i == 0:
#             continue
#         if i % 2:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].left = tmp
#             stack.append(tmp)
#         else:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].right = tmp
#             stack.append(tmp)
#     return root
#
#
# def print_tree(root: TreeNode):
#     res = []
#     queue = [root]
#     while queue:
#         tmp = queue.pop(0)
#         res.append(tmp.val)
#         if tmp.left:
#             queue.append(tmp.left)
#         if tmp.right:
#             queue.append(tmp.right)
#
#     print(res)


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


x = 0
# [
#   [1,null,3,2],
#   [3,2,null,1],
#   [3,1,null,null,2],
#   [2,1,3],
#   [1,null,2,null,3]
# ]


class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        # 两边可以取到
        @functools.lru_cache(None)
        def generate_subtree(head, tail):
            if head > tail:
                return [None]
            elif head == tail:
                return [TreeNode(head)]
            res = []
            for ele in range(head, tail + 1):
                for left in generate_subtree(head, ele - 1):
                    for right in generate_subtree(ele + 1, tail):
                        root = TreeNode(ele)
                        root.left = left
                        root.right = right
                        res.append(root)

            return res

        tmp = generate_subtree(1, n)
        return tmp


        # def generate_trees(start, end):
        #     if start > end:
        #         return [None, ]
        #
        #     all_trees = []
        #     for i in range(start, end + 1):  # pick up a root
        #         # all possible left subtrees if i is choosen to be a root
        #         left_trees = generate_trees(start, i - 1)
        #
        #         # all possible right subtrees if i is choosen to be a root
        #         right_trees = generate_trees(i + 1, end)
        #
        #         # connect left and right subtrees to the root i
        #         for l in left_trees:
        #             for r in right_trees:
        #                 current_tree = TreeNode(i)
        #                 current_tree.left = l
        #                 current_tree.right = r
        #                 all_trees.append(current_tree)
        #
        #     return all_trees
        #
        # return generate_trees(1, n) if n else []


solve = Solution()
roots = solve.generateTrees(x)
for root in roots:
    print(treeNodeToString(root))

# ############################### 96. 不同的二叉搜索树 ###############################
#
#
# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.right = None
#
#
# x = 1
# # [
# #   [1,null,3,2],
# #   [3,2,null,1],
# #   [3,1,null,null,2],
# #   [2,1,3],
# #   [1,null,2,null,3]
# # ]     5
#
#
# class Solution(object):
#     def numTrees(self, n):
#         """
#         :type n: int
#         :rtype: int
#         """
#         if n < 1:
#             return 0
#         dp = [1 for _ in range(n+1)]
#         for i in range(2, n + 1):
#             tmp = 0
#             ii = 0
#             jj = i - 1
#             while ii <= i - 1:
#                 tmp += dp[ii] * dp[jj]
#                 ii += 1
#                 jj -= 1
#             dp[i] = tmp
#
#         return dp[-1]
#
#     # def numTrees(self, n: int) -> int:
#     #     dp = [0] * (n + 1)
#     #     dp[0] = 1
#     #     dp[1] = 1
#     #
#     #     for i in range(2, n + 1):
#     #         for j in range(i):
#     #             dp[i] += dp[j] * dp[i - j - 1]
#     #
#     #     return dp[-1]
#
#
# solve = Solution()
# print(solve.numTrees(x))

# ############################### 97. 交错字符串 ###############################
# s1 = "aabcc"
# s2 = "dbbca"
# s3 = "aadbbcbcac"       # True
#
# # s1 = "aacc"
# # s2 = "dbbca"
# # s3 = "aadbbccac"       # True
#
# # s1 = "aabcc"
# # s2 = "dbbca"
# # s3 = "aadbbbaccc"       # False
# #
# # s1 = "a"
# # s2 = "b"
# # s3 = "a"       # False
#
# # s1 = "db"
# # s2 = "b"
# # s3 = "cbb"      # False
#
#
# class Solution(object):
#     def isInterleave(self, s1, s2, s3):
#         """
#         :type s1: str
#         :type s2: str
#         :type s3: str
#         :rtype: bool
#         """
#         # if len(s3) != len(s2) + len(s1):
#         #     return False
#         # dp = [[True] * (len(s2)+1) for _ in range((len(s1)+1))]
#         # for j in range(1, len(s2) + 1):
#         #     dp[0][j] = (s2[j-1] == s3[j-1]) and dp[0][j - 1]
#         # for i in range(1, len(s1) + 1):
#         #     dp[i][0] = s1[i-1] == s3[i-1] and dp[i - 1][0]
#         #
#         # for j in range(1, len(s2) + 1):
#         #     for i in range(1, len(s1) + 1):
#         #         if dp[i][j-1] and s2[j-1] == s3[i+j-1]:
#         #             dp[i][j] = True
#         #         elif dp[i-1][j] and s1[i-1] == s3[i+j-1]:
#         #             dp[i][j] = True
#         #         else:
#         #             dp[i][j] = False
#         #
#         # return dp[-1][-1]
#
#         # BFS
#         from collections import deque
#         n1 = len(s1)
#         n2 = len(s2)
#         n3 = len(s3)
#         if n1 + n2 != n3: return False
#
#         queue = deque()
#         queue.appendleft((0, 0))
#         visited = set()
#         while queue:
#             i, j = queue.pop()
#             if i == n1 and j == n2:
#                 return True
#             if i < n1 and s1[i] == s3[i + j] and (i + 1, j) not in visited:
#                 visited.add((i + 1, j))
#                 queue.appendleft((i + 1, j))
#             if j < n2 and s2[j] == s3[i + j] and (i, j + 1) not in visited:
#                 visited.add((i, j + 1))
#                 queue.appendleft((i, j + 1))
#         return False
#
#
# solve = Solution()
# print(solve.isInterleave(s1, s2, s3))
#

# ############################### 98. 验证二叉搜索树 ###############################
#
#
# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
#
# def create_tree(lis):
#     root = TreeNode(lis[0])
#     stack = [root]
#     for i, ele in enumerate(lis):
#         if i == 0:
#             continue
#         if i % 2:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].left = tmp
#             stack.append(tmp)
#         else:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].right = tmp
#             stack.append(tmp)
#     return root
#
#
# def print_tree(root: TreeNode):
#     res = []
#     queue = [root]
#     while queue:
#         tmp = queue.pop(0)
#         res.append(tmp.val)
#         if tmp.left:
#             queue.append(tmp.left)
#         if tmp.right:
#             queue.append(tmp.right)
#
#     print(res)
#
#
# x = create_tree([2,1,3])
# x = create_tree([5,1,4,None,None,3,6])
#
#
# class Solution(object):
#     def isValidBST(self, root):
#         """
#         :type root: TreeNode
#         :rtype: bool
#         """
#         # 利用二叉搜索树中序遍历是递增的
#         res = []
#
#         def helper(root):
#             if not root:
#                 return
#             helper(root.left)
#             res.append(root.val)
#             helper(root.right)
#
#         helper(root)
#         return res == sorted(res) and len(set(res)) == len(res)
#
#
#         # # 逐个子树比较的方法
#         # p = root
#         # pp = root
#         # if root:
#         #     if root.left:
#         #         if root.left.val <= root.val:
#         #             if not self.isValidBST(root.left):
#         #                 return False
#         #         else:
#         #             return False
#         #     if root.right:
#         #         if root.right.val >= root.val:
#         #             if not self.isValidBST(root.right):
#         #                 return False
#         #         else:
#         #             return False
#         # if not p:
#         #     return True
#         # tmp = p.val
#         # if p.right:
#         #     p = p.right
#         #     while p.left:
#         #         p = p.left
#         #     if p.val <= tmp:
#         #         return False
#         # if pp.left:
#         #     pp = pp.left
#         #     while pp.right:
#         #         pp = pp.right
#         #     if pp.val >= tmp:
#         #         return False
#         # return True
#
#
# solve = Solution()
# print(solve.isValidBST(x))

# ############################### 99. 恢复二叉搜索树 ###############################
#
#
# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
#
# def create_tree(lis):
#     root = TreeNode(lis[0])
#     stack = [root]
#     for i, ele in enumerate(lis):
#         if i == 0:
#             continue
#         if i % 2:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].left = tmp
#             stack.append(tmp)
#         else:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].right = tmp
#             stack.append(tmp)
#     return root
#
#
# def print_tree(root: TreeNode):
#     res = []
#     queue = [root]
#     while queue:
#         tmp = queue.pop(0)
#         res.append(tmp.val)
#         if tmp.left:
#             queue.append(tmp.left)
#         if tmp.right:
#             queue.append(tmp.right)
#
#     print(res)
#
#
# def prettyPrintTree(node, prefix="", isLeft=True):
#     if not node:
#         print("Empty Tree")
#         return
#
#     if node.right:
#         prettyPrintTree(node.right, prefix + ("│   " if isLeft else "    "), False)
#
#     print(prefix + ("└── " if isLeft else "┌── ") + str(node.val))
#
#     if node.left:
#         prettyPrintTree(node.left, prefix + ("    " if isLeft else "│   "), True)
#
#
# def treeNodeToString(root):
#     if not root:
#         return "[]"
#     output = ""
#     queue = [root]
#     current = 0
#     while current != len(queue):
#         node = queue[current]
#         current = current + 1
#
#         if not node:
#             output += "null, "
#             continue
#
#         output += str(node.val) + ", "
#         queue.append(node.left)
#         queue.append(node.right)
#     return "[" + output[:-2] + "]"
#
#
# def stringToTreeNode(input):    # stringToTreeNode('[1,null,2]')
#     input = input.strip()
#     input = input[1:-1]
#     if not input:
#         return None
#
#     inputValues = [s.strip() for s in input.split(',')]
#     root = TreeNode(int(inputValues[0]))
#     nodeQueue = [root]
#     front = 0
#     index = 1
#     while index < len(inputValues):
#         node = nodeQueue[front]
#         front = front + 1
#
#         item = inputValues[index]
#         index = index + 1
#         if item != "null":
#             leftNumber = int(item)
#             node.left = TreeNode(leftNumber)
#             nodeQueue.append(node.left)
#
#         if index >= len(inputValues):
#             break
#
#         item = inputValues[index]
#         index = index + 1
#         if item != "null":
#             rightNumber = int(item)
#             node.right = TreeNode(rightNumber)
#             nodeQueue.append(node.right)
#     return root
#
#
# x = stringToTreeNode('[1,3,null,null,2]')
# x = stringToTreeNode('[3,2,null,null,1]')
# # x = stringToTreeNode('[3,1,4,null,null,2]')
#
#
# class Solution(object):
#     def recoverTree(self, root):
#         """
#         :type root: TreeNode
#         :rtype: None Do not return anything, modify root in-place instead.
#         """
#         # Morris Traversal
#         cur = root
#         res = []
#
#         while cur:
#             if not cur.left:
#                 res.append(cur.val)
#                 cur = cur.right
#             else:
#                 pre = cur.left
#                 while pre.right != cur and pre.right:
#                     pre = pre.right
#
#                 if not pre.right:
#                     pre.right = cur
#                     cur = cur.left
#                 else:
#                     pre.right = None
#                     res.append(cur.val)
#                     cur = cur.right
#         print(res)
#         return res
#
#         # # 利用在类内部的性质简写的版本
#         # # 但是其实不满足常数空间复杂度,需要用莫里斯遍历,利用叶子节点左右子树为空,将根节点记录为叶子节点的后继
#         # self.firstNode = None
#         # self.secondNode = None
#         # self.preNode = TreeNode(float("-inf"))
#         #
#         # def in_order(root):
#         #     if not root:
#         #         return
#         #     in_order(root.left)
#         #     if self.firstNode == None and self.preNode.val >= root.val:
#         #         self.firstNode = self.preNode
#         #     if self.firstNode and self.preNode.val >= root.val:
#         #         self.secondNode = root
#         #     self.preNode = root
#         #     in_order(root.right)
#         #
#         # in_order(root)
#         # self.firstNode.val, self.secondNode.val = self.secondNode.val, self.firstNode.val
#
#
#         # 利用二叉搜索树的中序遍历递增
#         # rem = [TreeNode(float("-inf"))]
#         # res = [None, None]
#         #
#         # def mid_search(p):
#         #     if res[0] and res[1]:
#         #         return
#         #     if not p:
#         #         return
#         #     mid_search(p.left)
#         #     if not res[0]:
#         #         if p.val < rem[0].val:
#         #             res[0] = rem[0]
#         #             # res.append(rem[0])
#         #         rem[0] = p
#         #     elif not res[1]:
#         #         if p.val > res[0].val:
#         #             res[1] = rem[0]
#         #             return
#         #         rem[0] = p
#         #
#         #     mid_search(p.right)
#         #
#         # mid_search(root)
#         #
#         # if not res[1]:
#         #     res[1] = rem[0]
#         # print(res[0].val)
#         # print(res[1].val)
#         # res[0].val, res[1].val = res[1].val, res[0].val
#
#
# solve = Solution()
# solve.recoverTree(x)
# print(treeNodeToString(x))

# ############################### 100. 相同的树 ###############################
#
#
# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
#
# def create_tree(lis):
#     root = TreeNode(lis[0])
#     stack = [root]
#     for i, ele in enumerate(lis):
#         if i == 0:
#             continue
#         if i % 2:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].left = tmp
#             stack.append(tmp)
#         else:
#             tmp = TreeNode(ele)
#             stack[(i-1) // 2].right = tmp
#             stack.append(tmp)
#     return root
#
#
# def print_tree(root: TreeNode):
#     res = []
#     queue = [root]
#     while queue:
#         tmp = queue.pop(0)
#         res.append(tmp.val)
#         if tmp.left:
#             queue.append(tmp.left)
#         if tmp.right:
#             queue.append(tmp.right)
#
#     print(res)
#
#
# x = create_tree([1,2])
# y = create_tree([1,None,2])
#
#
# class Solution(object):
#     def isSameTree(self, p, q):
#         """
#         :type p: TreeNode
#         :type q: TreeNode
#         :rtype: bool
#         """
#         if not (p and q):
#             if p or q:
#                 return False
#             else:
#                 return True
#         if p.val == q.val:
#             return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
#         else:
#             return False
#
#
# solve = Solution()
# print(solve.isSameTree(x,y))
