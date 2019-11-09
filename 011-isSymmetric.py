import copy
import time
# ############################### 101. 对称二叉树 ###############################


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


# x = stringToTreeNode('[1,2,2,3,4,4,3]')
# x = stringToTreeNode('[1,2,2,null,3,null,3]')
# x = stringToTreeNode('[1,2,2,2,null,2]')
# x = stringToTreeNode('[1,2,2,3,4,4,3,5,6,7,8,8,7,6,5]')
# x = stringToTreeNode('[]')
#
#
# class Solution(object):
#     def isSymmetric(self, root):
#         """
#         :type root: TreeNode
#         :rtype: bool
#         """
#         # 利用中序遍历对称--不行,反例: [1,2,2,2,null,2]
#         # 改为递归,每次比较左子树和右子树
#         def compare(root1, root2):
#             if not (root1 or root2):
#                 return True
#             elif not (root1 and root2):
#                 return False
#             elif root1.val == root2.val:
#                 return compare(root1.left, root2.right) and compare(root2.left, root1.right)
#             else:
#                 return False
#
#         if not root:
#             return True
#         return compare(root.left, root.right)
#
#         # 栈,迭代
#         # if not root: return True
#         #
#         # def Tree(p, q):
#         #     stack = [(q, p)]
#         #     while stack:
#         #         a, b = stack.pop()
#         #         if not a and not b:
#         #             continue
#         #         if a and b and a.val == b.val:
#         #             stack.append((a.left, b.right))
#         #             stack.append((a.right, b.left))
#         #         else:
#         #             return False
#         #     return True
#         #
#         # return Tree(root.left, root.right)
#
#
# solve = Solution()
# print(solve.isSymmetric(x))

# ############################### 102. 二叉树的层次遍历 ###############################
# x = stringToTreeNode('[3,9,20,null,null,15,7]')
# # [
# #   [3],
# #   [9,20],
# #   [15,7]
# # ]
#
#
# class Solution(object):
#     def levelOrder(self, root):
#         """
#         :type root: TreeNode
#         :rtype: List[List[int]]
#         """
#         # def read_level(roots):
#         #     tmp = []
#         #     res = []
#         #     for root in roots:
#         #         if root.left:
#         #             res.append(root.left)
#         #             tmp.append(root.left.val)
#         #         if root.right:
#         #             res.append(root.right)
#         #             tmp.append(root.right.val)
#         #     if not res:
#         #         return
#         #     ans.append(tmp)
#         #     read_level(res)
#         #
#         # if not root:
#         #     return []
#         # ans = []
#         # ans.append([root.val])
#         # read_level([root])
#         # return ans
#
#         # 这么写不用存中间节点
#         res = []
#
#         def helper(root, depth):
#             if not root: return
#             if len(res) == depth:
#                 res.append([])
#             res[depth].append(root.val)
#             helper(root.left, depth + 1)
#             helper(root.right, depth + 1)
#
#         helper(root, 0)
#         return res
#
#
# solve = Solution()
# print(solve.levelOrder(x))

# ############################### 103. 二叉树的锯齿形层次遍历 ###############################
# 即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行
# x = stringToTreeNode('[3,9,20,null,null,15,7]')
# # [
# #   [3],
# #   [20,9],
# #   [15,7]
# # ]
# # x = stringToTreeNode('[1,2,3,4,5,6,7]')
# x = stringToTreeNode('[0,2,4,1,null,3,-1,5,1,null,6,null,8]')
#
#
# class Solution(object):
#     def zigzagLevelOrder(self, root):
#         """
#         :type root: TreeNode
#         :rtype: List[List[int]]
#         """
#         # def read_level(roots, depth):
#         #     tmp = []
#         #     res = []
#         #     for root in roots:
#         #         if root.left:
#         #             res.append(root.left)
#         #             tmp.append(root.left.val)
#         #         if root.right:
#         #             res.append(root.right)
#         #             tmp.append(root.right.val)
#         #     if not res:
#         #         return
#         #     if not depth % 2:
#         #         ans.append(tmp[::-1])
#         #     else:
#         #         ans.append(tmp)
#         #     read_level(res, depth + 1)
#         #
#         # if not root:
#         #     return []
#         # ans = []
#         # ans.append([root.val])
#         # read_level([root], 0)
#         # return ans
#
#         # 上个方法的改进
#         res = []
#
#         def helper(root, depth):
#             if not root: return
#             if len(res) == depth:
#                 res.append([])
#             if depth % 2 == 0:
#                 res[depth].append(root.val)
#             else:
#                 res[depth].insert(0, root.val)
#             helper(root.left, depth + 1)
#             helper(root.right, depth + 1)
#
#         helper(root, 0)
#         return res
#
#
# solve = Solution()
# print(solve.zigzagLevelOrder(x))

# ############################### 104. 二叉树的最大深度 ###############################
# x = stringToTreeNode('[3,9,20,null,null,15,7,null,null,null,1]')     # 3
#
#
# class Solution(object):
#     def maxDepth(self, root):
#         """
#         :type root: TreeNode
#         :rtype: int
#         """
#         # length = [0]
#         # if not root:
#         #     return 0
#         #
#         # def find_depth(root, depth):
#         #     if root.left:
#         #         find_depth(root.left, depth + 1)
#         #     if root.right:
#         #         find_depth(root.right, depth + 1)
#         #     if not (root.left or root.right):
#         #         if depth > length[0]:
#         #             length[0] = depth
#         #
#         # find_depth(root, 1)
#         # return length[0]
#
#         # # 简化递归
#         # if not root: return 0
#         # left, right = self.maxDepth(root.left), self.maxDepth(root.right)
#         # return max(left, right) + 1
#
#         # 用迭代
#         from collections import deque
#         if not root: return 0
#         queue = deque()
#         queue.appendleft(root)
#         res = 0
#         while queue:
#             # print(queue)
#             res += 1
#             n = len(queue)
#             for _ in range(n):
#                 tmp = queue.pop()
#                 if tmp.left:
#                     queue.appendleft(tmp.left)
#                 if tmp.right:
#                     queue.appendleft(tmp.right)
#         return res
#
#
# solve = Solution()
# print(solve.maxDepth(x))

# ############################ 105. 从前序与中序遍历序列构造二叉树 ###########################
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]     # [3,9,20,null,null,15,7]

# preorder = []
# inorder = []     # [3,9,20,null,null,15,7]


class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # # 这内存占用巨大
        # def create_tree(preorder, inorder):
        #     if not preorder:
        #         return
        #     tmp = preorder[0]
        #     root = TreeNode(tmp)
        #     ind = inorder.index(tmp)
        #     root.left = create_tree(preorder[1:ind+1], inorder[:ind])
        #     root.right = create_tree(preorder[ind + 1:], inorder[ind + 1:])
        #     return root
        #
        # return create_tree(preorder, inorder)

        # from collections import defaultdict
        # n = len(preorder)
        # inorder_map = defaultdict(int)
        # for idx, val in enumerate(inorder):
        #     inorder_map[val] = idx
        #
        # def helper(pre_start, pre_end, in_start, in_end):
        #     if pre_start == pre_end:
        #         return None
        #     root = TreeNode(preorder[pre_start])
        #     loc = inorder_map[preorder[pre_start]]
        #     # 这里要注意 因为 一开始可以明确是 pre_start + 1,in_start,loc,因为前序和中序个数是相同,所以可以求出前序左右边界
        #     root.left = helper(pre_start + 1, pre_start + 1 + loc - in_start, in_start, loc)
        #     # 根据上面用过的, 写出剩下就行了
        #     root.right = helper(pre_start + 1 + loc - in_start, pre_end, loc + 1, in_end)
        #     return root
        #
        # return helper(0, n, 0, n)

        if len(preorder) == 0:
            return None
        root = TreeNode(preorder[0])
        stack = []
        stack.append(root)
        iidx = 0

        for pidx in range(1, len(preorder)):
            pnode = stack[-1]
            cnode = TreeNode(preorder[pidx])

            if pnode.val != inorder[iidx]:
                pnode.left = cnode
                stack.append(cnode)
                continue

            while stack and stack[-1].val == inorder[iidx]:
                pnode = stack[-1]
                stack.pop()
                iidx = iidx + 1
            pnode.right = cnode
            stack.append(cnode)
        return root


solve = Solution()
print(treeNodeToString(solve.buildTree(preorder, inorder)))
