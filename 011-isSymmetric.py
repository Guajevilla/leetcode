import copy
import time
import json
# ############################### 101. 对称二叉树 ###############################


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
# preorder = [3,9,20,15,7]
# inorder = [9,3,15,20,7]     # [3,9,20,null,null,15,7]
#
# # preorder = []
# # inorder = []     # [3,9,20,null,null,15,7]
#
#
# class Solution(object):
#     def buildTree(self, preorder, inorder):
#         """
#         :type preorder: List[int]
#         :type inorder: List[int]
#         :rtype: TreeNode
#         """
#         # # 这内存占用巨大
#         # def create_tree(preorder, inorder):
#         #     if not preorder:
#         #         return
#         #     tmp = preorder[0]
#         #     root = TreeNode(tmp)
#         #     ind = inorder.index(tmp)
#         #     root.left = create_tree(preorder[1:ind+1], inorder[:ind])
#         #     root.right = create_tree(preorder[ind + 1:], inorder[ind + 1:])
#         #     return root
#         #
#         # return create_tree(preorder, inorder)
#
#         # from collections import defaultdict
#         # n = len(preorder)
#         # inorder_map = defaultdict(int)
#         # for idx, val in enumerate(inorder):
#         #     inorder_map[val] = idx
#         #
#         # def helper(pre_start, pre_end, in_start, in_end):
#         #     if pre_start == pre_end:
#         #         return None
#         #     root = TreeNode(preorder[pre_start])
#         #     loc = inorder_map[preorder[pre_start]]
#         #     # 这里要注意 因为 一开始可以明确是 pre_start + 1,in_start,loc,因为前序和中序个数是相同,所以可以求出前序左右边界
#         #     root.left = helper(pre_start + 1, pre_start + 1 + loc - in_start, in_start, loc)
#         #     # 根据上面用过的, 写出剩下就行了
#         #     root.right = helper(pre_start + 1 + loc - in_start, pre_end, loc + 1, in_end)
#         #     return root
#         #
#         # return helper(0, n, 0, n)
#
#         if len(preorder) == 0:
#             return None
#         root = TreeNode(preorder[0])
#         stack = []
#         stack.append(root)
#         iidx = 0
#
#         for pidx in range(1, len(preorder)):
#             pnode = stack[-1]
#             cnode = TreeNode(preorder[pidx])
#
#             if pnode.val != inorder[iidx]:
#                 pnode.left = cnode
#                 stack.append(cnode)
#                 continue
#
#             while stack and stack[-1].val == inorder[iidx]:
#                 pnode = stack[-1]
#                 stack.pop()
#                 iidx = iidx + 1
#             pnode.right = cnode
#             stack.append(cnode)
#         return root
#
#
# solve = Solution()
# print(treeNodeToString(solve.buildTree(preorder, inorder)))

# ############################ 106. 从中序与后序遍历序列构造二叉树 ###########################
inorder = [9,3,15,20,7]
postorder = [9,15,7,20,3]       # [3,9,20,null,null,15,7]


class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        # # 同上题,内存占用巨大
        # def create_tree(inorder, postorder):
        #     if not postorder:
        #         return
        #     tmp = postorder[-1]
        #     root = TreeNode(tmp)
        #     ind = inorder.index(tmp)
        #     root.left = create_tree(inorder[:ind], postorder[:ind])
        #     root.right = create_tree(inorder[ind + 1:], postorder[ind:-1])
        #     return root
        #
        # return create_tree(inorder, postorder)

        # from collections import defaultdict
        # n = len(inorder)
        # inorder_map = defaultdict(int)
        # for idx, val in enumerate(inorder):
        #     inorder_map[val] = idx
        # #print(inorder_map)
        # def helper(in_start, in_end, post_start, post_end):
        #     if in_start == in_end:
        #         return None
        #     #print(post_end)
        #     root = TreeNode(postorder[post_end - 1])
        #     loc = inorder_map[postorder[post_end - 1]]
        #     root.left = helper(in_start, loc, post_start, post_start + loc - in_start)
        #     root.right = helper(loc + 1, in_end, post_start + loc - in_start, post_end - 1)
        #     return root
        #
        # return helper(0, n, 0, n)

        n = len(inorder)
        if n == 0:
            return None
        iidx = pidx = n - 1
        root = TreeNode(postorder[pidx])
        pidx = pidx - 1
        stack = [root]
        while pidx >= 0:
            pnode = stack[-1]
            cnode = TreeNode(postorder[pidx])
            if pnode.val != inorder[iidx]:
                pnode.right = cnode
                stack.append(cnode)
                pidx = pidx - 1
                continue
            while stack and stack[-1].val == inorder[iidx]:
                pnode = stack[-1]
                stack.pop()
                iidx = iidx - 1

            pnode.left = cnode
            stack.append(cnode)
            pidx = pidx - 1
        return root


solve = Solution()
print(treeNodeToString(solve.buildTree(inorder, postorder)))

# ############################ 107. 二叉树的层次遍历 II ###########################
# x = stringToTreeNode('[3,9,20,null,null,15,7]')
# # [
# #   [15,7],
# #   [9,20],
# #   [3]
# # ]
#
#
# class Solution(object):
#     def levelOrderBottom(self, root):
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
#         #     ans.insert(0, tmp)
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
#                 res.insert(0, [])
#             res[-(depth+1)].append(root.val)
#             helper(root.left, depth + 1)
#             helper(root.right, depth + 1)
#
#         helper(root, 0)
#         return res
#
#
# solve = Solution()
# print(solve.levelOrderBottom(x))

# ############################ 108. 将有序数组转换为二叉搜索树 ###########################
# # 还要求高度平衡
# nums = [-10,-3,0,5,9]       # [0,-3,9,-10,null,5]
#
#
# class Solution(object):
#     def sortedArrayToBST(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: TreeNode
#         """
#         def create_subtree(nums):
#             if not nums:
#                 return
#             length = len(nums) // 2
#             root = TreeNode(nums[length])
#             root.left = create_subtree(nums[:length])
#             root.right = create_subtree(nums[length+1:])
#             return root
#
#         return create_subtree(nums)
#
#
# solve = Solution()
# print(treeNodeToString(solve.sortedArrayToBST(nums)))

# ############################ 109. 有序链表转换二叉搜索树 ###########################
# head = stringToListNode('[-10, -3, 0, 5, 9]')          # [0,-3,9,-10,null,5]
# # head = stringToListNode('[]')          # [0,-3,9,-10,null,5]
#
#
# class Solution(object):
#     # # 利用快慢指针找中点1
#     # def sortedListToBST(self, head):
#     #     """
#     #     :type head: ListNode
#     #     :rtype: TreeNode
#     #     """
#     #     def find_mid(head):
#     #         dummy = ListNode(0)
#     #         dummy.next = head
#     #         pre = dummy
#     #         slow = head
#     #         fast = head
#     #         while fast.next:
#     #             slow = slow.next
#     #             fast = fast.next
#     #             pre = pre.next
#     #             if fast.next:
#     #                 fast = fast.next
#     #             else:
#     #                 break
#     #
#     #         pre.next = None
#     #         return slow
#     #
#     #     def create_subtree(head):
#     #         if not head:
#     #             return
#     #         elif not head.next:
#     #             return TreeNode(head.val)
#     #         mid = find_mid(head)
#     #         root = TreeNode(mid.val)
#     #         root.left = create_subtree(head)
#     #         root.right = create_subtree(mid.next)
#     #         return root
#     #
#     #     return create_subtree(head)
#
#
#     # 利用快慢指针找中点2
#     def sortedListToBST(self, head: ListNode) -> TreeNode:
#         def findmid(head, tail):
#             slow = head
#             fast = head
#             while fast != tail and fast.next != tail:
#                 slow = slow.next
#                 fast = fast.next.next
#             return slow
#
#         def helper(head, tail):
#             if head == tail: return
#             node = findmid(head, tail)
#             root = TreeNode(node.val)
#             root.left = helper(head, node)
#             root.right = helper(node.next, tail)
#             return root
#
#         return helper(head, None)
#
#
#     # # 先转化为列表,后利用列表索引
#     # def sortedListToBST(self, head):
#     #     """
#     #     :type head: ListNode
#     #     :rtype: TreeNode
#     #     """
#     #     rem = []
#     #     while head:
#     #         rem.append(head.val)
#     #         head = head.next
#     #
#     #     def create_subtree(nums):
#     #         if not nums:
#     #             return
#     #         length = len(nums) // 2
#     #         root = TreeNode(nums[length])
#     #         root.left = create_subtree(nums[:length])
#     #         root.right = create_subtree(nums[length+1:])
#     #         return root
#     #
#     #     return create_subtree(rem)
#
#
# solve = Solution()
# print(treeNodeToString(solve.sortedListToBST(head)))

# ############################ 110. 平衡二叉树 ###########################
x = stringToTreeNode('[3,9,20,null,null,15,7]')     # True
# x = stringToTreeNode('[1,2,2,3,3,null,null,4,4]')     # F
x = stringToTreeNode('[1]')     # True[1,2,2,3,3,3,3,4,4,4,4,4,4,null,null,5,5]
# x = stringToTreeNode('[1,2,2,3,3,3,3,4,4,4,4,4,4,null,null,5,5]')     # True
# x = stringToTreeNode('[7,0,3,8,5,7,9,3,2,2,7,4,1,7,0,6,3,0,4,7,8,0,4,5,4,4,1,3,7,3,7,9,5,6,6,7,9,5,5,8,6,2,2,6,2,5,3,8,5,6,1,0,8,6,3,1,4,2,6,4,9,5,0,0,2,1,5,7,3,8,2,2,8,2,2,4,2,2,2,5,5,5,7,0,2,7,6,3,2,7,6,5,9,6,2,7,5,8,0,6,8,1,4,0,4,8,8,1,8,5,3,6,7,5,3,0,1,8,8,2,0,3,9,3,2,1,8,3,8,8,3,9,9,5,1,1,2,7,2,6,5,8,1,3,2,8,0,2,8,9,8,1,7,1,8,1,5,2,8,8,8,0,8,3,5,8,7,1,2,8,0,3,2,8,7,6,9,5,9,1,6,7,9,9,3,6,3,0,4,3,8,1,9,8,0,4,8,5,3,4,1,7,8,3,7,4,7,3,1,0,7,7,5,8,2,6,1,5,3,3,5,4,7,9,9,0,9,4,4,8,5,4,3,7,8,3,7,6,0,3,0,9,8,5,7,4,4,0,7,7,8,7,3,4,9,5,3,6,9,7,4,7,1,7,null,3,3,0,9,0,1,4,0,7,2,1,6,6,9,4,8,3,0,5,5,0,8,9,8,5,null,8,5,5,1,9,8,9,0,2,4,6,8,8,1,8,4,9,2,5,9,1,6,7,5,0,0,0,2,6,8,3,3,0,7,1,7,8,2,3,2,9,9,2,7,3,5,0,9,0,2,3,4,0,4,2,0,0,4,0,1,3,1,1,9,1,1,8,9,1,0,0,9,9,3,4,8,0,2,7,6,8,3,2,6,6,2,9,1,2,1,7,6,6,5,0,9,5,8,7,5,9,1,4,6,2,9,7,3,7,8,0,8,2,4,9,9,0,9,1,8,7,3,2,6,4,8,9,3,5,7,4,0,8,1,0,1,7,3,4,2,8,4,5,2,8,5,0,3,0,5,7,3,9,2,0,8,5,7,4,7,3,6,3,7,9,5,9,2,4,9,9,6,5,5,4,6,5,2,6,6,3,8,0,9,6,2,6,7,0,2,1,5,3,5,3,9,9,0,7,9,9,5,3,2,0,4,4,1,6,1,4,7,1,1,4,6,1,3,8,7,7,2,9,4,8,6,6,1,5,7,0,2,null,null,null,null,null,null,5,2,8,4,2,5,6,5,0,8,7,1,1,9,9,null,4,4,1,null,8,4,8,5,5,4,5,5,5,6,0,4,5,3,1,3,5,4,1,2,9,9,8,7,7,9,8,5,null,null,3,2,3,7,2,4,8,null,null,null,null,null,4,7,4,3,6,6,3,8,6,9,4,1,7,8,9,8,2,5,9,8,5,7,9,8,0,2,8,null,7,1,2,null,1,6,2,7,8,7,3,6,9,6,3,null,7,2,8,9,7,8,9,4,4,8,1,3,7,8,4,null,7,8,7,8,8,null,null,null,1,5,6,4,0,null,4,null,8,7,1,3,1,1,8,7,1,4,9,null,7,4,2,null,1,0,5,0,2,9,3,2,8,3,0,null,null,null,null,null,6,5,9,1,2,9,7,6,8,7,7,0,0,3,2,1,8,2,0,4,9,8,7,1,6,6,7,null,7,null,6,null,3,7,7,9,0,2,4,null,1,null,null,null,0,1,9,6,1,1,7,9,7,8,4,null,8,2,4,3,4,7,3,null,5,null,7,null,5,null,5,null,2,5,9,6,3,1,6,5,0,7,1,2,0,8,3,7,2,3,9,9,1,4,2,null,2,null,5,null,6,null,7,null,0,null,6,2,3,7,9,null,7,0,4,2,2,9,2,1,9,null,null,null,9,2,3,9,0,7,5,null,9,8,2,9,0,0,6,9,3,3,6,2,3,null,3,null,2,9,9,2,3,7,9,null,7,5,0,null,9,2,6,3,7,null,null,null,8,1,4,1,null,null,null,null,7,7,4,9,1,2,2,0,1,1,7,3,5,5,7,null,4,6,7,9,4,7,2,4,6,null,7,2,5,5,3,null,7,2,7,2,7,null,3,null,8,7,4,3,4,6,4,8,0,8,8,1,1,0,7,3,4,8,8,null,9,4,4,null,5,null,9,null,5,4,5,6,0,7,9,1,9,5,8,null,4,3,7,null,7,9,9,null,4,9,6,3,5,9,6,8,9,4,8,2,7,3,2,null,7,8,1,7,2,null,4,null,6,1,7,3,2,6,3,1,0,1,9,5,3,4,7,8,0,null,7,null,0,null,null,null,6,7,3,6,0,null,6,null,null,null,null,null,null,null,null,null,2,null,3,null,4,null,null,null,9,null,null,null,2,null,null,null,4,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,5,null,null,null,0,null,null,null,5,null,null,null,null,null,null,null,0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,3,null,null,null,null,null,null,null,5,null,8,null,2,null,null,null,null,null,null,null,null,null,6,null,null,null,5,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,9,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,0,1,7,6,null,null,null,null,null,null,7,null,null,null,8,null,null,null,null,null,null,null,9,null,null,null,9,null,null,null,7,null,null,null,8,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,6,null,null,null,null,null,null,null,2,null,null,null,null,null,null,null,null,null,null,null,null,null,7,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,3,null,null,null,0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,3,null,4,null,7,null,6,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,6,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,5,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,4,null,null,null,null,null,null,null,6,null,null,null,8,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,4,8,4,0,null,null,null,null,null,null,2,null,null,null,9,null,null,null,3,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,4,null,1,null,9,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,3,null,8,null,8,null,null,null,null,null,5,4,1,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,8,7,5,5,null,null,null,null,null,null,null,null,null,null,1,8,7,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,8,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,6,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,4,null,null,null,3,null,null,null,6,null,null,null,null,null,null,null,null,null,null,null,null,null,5,null,null,null,8]')
# F


class Solution(object):
    def subtree_depth(self, root):
        if not root:
            return 0
        left, right = self.subtree_depth(root.left), self.subtree_depth(root.right)
        return max(left, right) + 1

    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def subtree_balance(root):
            left_depth = self.subtree_depth(root.left)
            right_depth = self.subtree_depth(root.right)
            return abs(left_depth - right_depth) <= 1

        if not root:
            return True
        return subtree_balance(root) and self.isBalanced(root.left) and self.isBalanced(root.right)


    # # 理解错了..不是最大深度和最小深度..
    # def maxDepth(self, root):
    #     """
    #     :type root: TreeNode
    #     :rtype: int
    #     """
    #     length = [0, float('inf')]
    #     if not root:
    #         return [0, 0]
    #
    #     def find_depth(root, depth):
    #         if root.left:
    #             find_depth(root.left, depth + 1)
    #         else:
    #             if depth < length[1]:
    #                 length[1] = depth
    #         if root.right:
    #             find_depth(root.right, depth + 1)
    #         else:
    #             if depth < length[1]:
    #                 length[1] = depth
    #         if not (root.left or root.right):
    #             if depth > length[0]:
    #                 length[0] = depth
    #
    #     find_depth(root, 1)
    #     return length
    #
    # def isBalanced(self, root):
    #     """
    #     :type root: TreeNode
    #     :rtype: bool
    #     """
    #     length = self.maxDepth(root)
    #     return (length[0] - length[1]) <= 1


solve = Solution()
print(solve.isBalanced(x))
