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


# ############################### 231. 2的幂 #################################
# x = 1
# x = 16
# # x = 218
# # x = 0
#
#
# class Solution:
#     def isPowerOfTwo(self, n: int) -> bool:
#         # 转化为二进制表示只能是100..的形式
#         # if n == 1:
#         #     return True
#         # tmp = bin(n)[2:]
#         # if tmp[0] == '1' and int(tmp[1:]) == 0:
#         #     return True
#         # return False
#
#         # 等价于这句话..
#         return n > 0 and bin(n).count("1") == 1
#
#         # 同样的想法,位操作
#         return n > 0 and n & (n - 1) == 0
#
#         # # 暴力迭代
#         # if n == 0: return False
#         # while n % 2 == 0:
#         #     n //= 2
#         # return n == 1
#
#
# solve = Solution()
# print(solve.isPowerOfTwo(x))

# ############################### 232. 用栈实现队列 #################################
#
# class MyQueue:
#
#     def __init__(self):
#         """
#         Initialize your data structure here.
#         """
#         self.queue = []
#
#     def push(self, x: int) -> None:
#         """
#         Push element x to the back of queue.
#         """
#         tmp = []
#         while self.queue:
#             tmp.append(self.queue.pop(-1))
#         self.queue.append(x)
#         while tmp:
#             self.queue.append(tmp.pop(-1))
#
#     def pop(self) -> int:
#         """
#         Removes the element from in front of queue and returns that element.
#         """
#         return self.queue.pop(-1)
#
#     def peek(self) -> int:
#         """
#         Get the front element.
#         """
#         return self.queue[-1]
#
#     def empty(self) -> bool:
#         """
#         Returns whether the queue is empty.
#         """
#         return len(self.queue) == 0
#
#
# # Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(1)
# obj.push(2)
# print(obj.peek())
# print(obj.pop())
# print(obj.empty())

# ############################### 233. 数字 1 的个数 #################################
# x = 13          # 6  (1, 10, 11, 12, 13)
# x = 130         # 64
# x = 110         # 33
# # x = 111         # 36
# # x = 121         # 55
# # x = 1100        # 422
# # x = 1111        # 448
#
#
# class Solution:
#     # 找规律..
#     def countDigitOne(self, n: int) -> int:
#         if n <= 0:
#             return 0
#         res = 0
#         rem = n
#         i = 1
#         while n // 10 > 0:
#             tmp = n % 10
#             res += n // 10 * i
#             if tmp > 1:
#                 res += i
#             elif tmp == 1:
#                 res += rem % i + 1
#             n = n // 10
#             i *= 10
#         if n == 1:
#             res += rem - i + 1
#         else:
#             res += i
#         return res
#
#         # cnt, i = 0, 1
#         # while i <= n:  # i 依次个十百位的算，直到大于 n 为止。
#         #     cnt += n // (i * 10) * i + min(max(n % (i * 10) - i + 1, 0), i)
#         #     i *= 10
#         # return cnt
#
#
#         # if n <= 0:
#         #     return 0
#         # if n < 10:
#         #     return 1
#         # last = int(str(n)[1:])
#         # power = 10 ** (len(str(n)) - 1)
#         # high = int(str(n)[0])
#         # if high == 1:
#         #     return self.countDigitOne(last) + self.countDigitOne(power - 1) + last + 1
#         # else:
#         #     return power + high * self.countDigitOne(power - 1) + self.countDigitOne(last);
#
#
# solve = Solution()
# print(solve.countDigitOne(x))

# ############################### 234. 回文链表 #################################
# 要求O(n) 时间复杂度和 O(1) 空间复杂度
x = stringToListNode('[1,2,3]')
x = stringToListNode('[1]')
x = stringToListNode('[1,11,11,1]')
# x = stringToListNode('[1,3,2,4,3,2,1]')


# class Solution:
#     def reverse_list(self, dummy):
#         p = dummy.next
#         while p.next:
#             tmp = dummy.next
#             dummy.next = p.next
#             p.next = dummy.next.next
#             dummy.next.next = tmp
#
#     def isPalindrome(self, head: ListNode) -> bool:
#         # 双指针加反转
#         if not head or not head.next:
#             return True
#         dummy = ListNode(0)
#         dummy.next = head
#         fast = dummy
#         slow = dummy
#         while fast and fast.next:
#             fast = fast.next.next
#             slow = slow.next
#         print(slow.val)
#         self.reverse_list(slow)
#         printList(dummy)
#         printList(slow)
#         p = slow.next
#         while p:
#             if p.val != head.val:
#                 return False
#             p = p.next
#             head = head.next
#         return True
#
#
#         # # 数学计算法
#         # s1 = 0
#         # s2 = 0
#         # t = 1
#         #
#         # while head:
#         #     s1 = s1 * 10 + head.val
#         #     s2 = s2 + t * head.val
#         #     t = t * 10
#         #     head = head.next
#         #
#         # return s1 == s2
#
#
# solve = Solution()
# print(solve.isPalindrome(x))

# ############################## 235. 二叉搜索树的最近公共祖先 ################################
# x = stringToTreeNode('[6,2,8,0,4,7,9,null,null,3,5]')
# p = x.left
# q = x.right     # 6
#
# x = stringToTreeNode('[6,2,8,0,4,7,9,null,null,3,5]')
# p = x.left
# q = x.left.right     # 2
#
#
# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         if p.val < root.val and q.val < root.val:
#             return self.lowestCommonAncestor(root.left, p, q)
#         elif p.val > root.val and q.val > root.val:
#             return self.lowestCommonAncestor(root.right, p, q)
#
#         return root
#
#
# solve = Solution()
# print(solve.lowestCommonAncestor(x, p, q).val)

# ############################## 236. 二叉树的最近公共祖先 ################################
# x = stringToTreeNode('[3,5,1,6,2,0,8,null,null,7,4]')
# p = x.left
# q = x.right     # 3
#
# # x = stringToTreeNode('[3,5,1,6,2,0,8,null,null,7,4]')
# # p = x.left
# # q = x.left.right.right     # 5
#
#
# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         order = []
#         ind = [0, 0]
#
#         def in_order(root):
#             if not root:
#                 return
#             in_order(root.left)
#             order.append(root.val)
#             if root.val == p.val:
#                 ind[0] = len(order) - 1
#             elif root.val == q.val:
#                 ind[1] = len(order) - 1
#             in_order(root.right)
#
#         in_order(root)
#         ind.sort()
#         i = order.index(root.val)
#         while ind[0] > i or i > ind[1]:
#             if i > ind[1]:
#                 root = root.left
#             elif i < ind[0]:
#                 root = root.right
#             i = order.index(root.val)
#         return root
#
#
#
# class Solution1:
#     # 递归深度遍历,当左右子树和当前节点三个中只要有2个出现了p和q,就说明当前节点是最近公共祖先
#     # O(n)
#     def __init__(self):
#         # Variable to store LCA node.
#         self.ans = None
#
#     def lowestCommonAncestor(self, root, p, q):
#         """
#         :type root: TreeNode
#         :type p: TreeNode
#         :type q: TreeNode
#         :rtype: TreeNode
#         """
#         def recurse_tree(current_node):
#
#             # If reached the end of a branch, return False.
#             if not current_node:
#                 return False
#
#             # Left Recursion
#             left = recurse_tree(current_node.left)
#
#             # Right Recursion
#             right = recurse_tree(current_node.right)
#
#             # If the current node is one of p or q
#             mid = current_node == p or current_node == q
#
#             # If any two of the three flags left, right or mid become True.
#             if mid + left + right >= 2:
#                 self.ans = current_node
#
#             # Return True if either of the three bool values is True.
#             return mid or left or right
#
#         # Traverse the tree
#         recurse_tree(root)
#         return self.ans
#
#
# class Solution2:
#     # Three static flags to keep track of post-order traversal.
#
#     # Both left and right traversal pending for a node.
#     # Indicates the nodes children are yet to be traversed.
#     BOTH_PENDING = 2
#     # Left traversal done.
#     LEFT_DONE = 1
#     # Both left and right traversal done for a node.
#     # Indicates the node can be popped off the stack.
#     BOTH_DONE = 0
#
#     def lowestCommonAncestor(self, root, p, q):
#         """
#         :type root: TreeNode
#         :type p: TreeNode
#         :type q: TreeNode
#         :rtype: TreeNode
#         """
#         stack = [(root, Solution.BOTH_PENDING)]
#
#         # This flag is set when either one of p or q is found.
#         one_node_found = False
#
#         # This is used to keep track of LCA index.
#         LCA_index = -1
#
#         # We do a post order traversal of the binary tree using stack
#         while stack:
#             parent_node, parent_state = stack[-1]
#
#             # If the parent_state is not equal to BOTH_DONE,
#             # this means the parent_node can't be popped of yet.
#             if parent_state != Solution.BOTH_DONE:
#
#                 # If both child traversals are pending
#                 if parent_state == Solution.BOTH_PENDING:
#
#                     # Check if the current parent_node is either p or q.
#                     if parent_node == p or parent_node == q:
#
#                         # If one_node_found is set already, this means we have found both the nodes.
#                         if one_node_found:
#                             return stack[LCA_index][0]
#                         else:
#                             # Otherwise, set one_node_found to True,
#                             # to mark one of p and q is found.
#                             one_node_found = True
#
#                             # Save the current top index of stack as the LCA_index.
#                             LCA_index = len(stack) - 1
#
#                     # If both pending, traverse the left child first
#                     child_node = parent_node.left
#                 else:
#                     # traverse right child
#                     child_node = parent_node.right
#
#                 # Update the node state at the top of the stack
#                 # Since we have visited one more child.
#                 stack.pop()
#                 stack.append((parent_node, parent_state - 1))
#
#                 # Add the child node to the stack for traversal.
#                 if child_node:
#                     stack.append((child_node, Solution.BOTH_PENDING))
#             else:
#                 # If the parent_state of the node is both done,
#                 # the top node could be popped off the stack.
#
#                 # i.e. If LCA_index is equal to length of stack. Then we decrease LCA_index by 1.
#                 if one_node_found and LCA_index == len(stack) - 1:
#                     LCA_index -= 1
#                 stack.pop()
#
#         return None
#
#
# solve = Solution()
# print(solve.lowestCommonAncestor(x, p, q).val)

# ############################## 237. 删除链表中的节点 ################################
# class Solution:
#     def deleteNode(self, node):
#         """
#         :type node: ListNode
#         :rtype: void Do not return anything, modify node in-place instead.
#         """
#         node.val = node.next.val
#         node.next = node.next.next
#
#
# solve = Solution()
# print(solve.deleteNode(x))

# ############################## 238. 除自身以外数组的乘积 ################################
# # 说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
# x = [1,2,3,4]       # [24,12,8,6]
# # x = []       # [24,12,8,6]
#
#
# class Solution:
#     def productExceptSelf(self, nums):
#         n = len(nums)
#         rem_l = [1] * n
#         rem_r = [1] * n
#         for i in range(1, n):
#             rem_l[i] = rem_l[i - 1] * nums[i - 1]
#             rem_r[n - 1 - i] = rem_r[n - i] * nums[n - i]
#         for i in range(n):
#             rem_l[i] = rem_l[i] * rem_r[i]
#         return rem_l
#
#         # 借助res可省O(n)的空间复杂度
#         n = len(nums)
#         res = [0]*n
#         k = 1
#         for i in range(n):
#             res[i] = k
#             k = k*nums[i]
#         k = 1
#         for i in range(n-1, -1, -1):
#             res[i] *= k
#             k *= nums[i]
#         return res
#
#
# solve = Solution()
# print(solve.productExceptSelf(x))

# ############################## 239. 滑动窗口最大值 ################################
# 要求线性时间复杂度
nums = [1,3,-1,-3,5,3,6,7]
k = 3               # [3,3,5,5,6,7]

# nums = [1,-1]
# k = 1               # [3,3,5,5,6,7]

nums = [7,1,3,-1,-3,5,3,6]
k = 3               # [7,3,3,5,5,6]


class Solution:
    def maxSlidingWindow(self, nums, k):
        # # 维护滑动窗口,使得滑动窗口内部是递减的,这样每次第一个数就是所求
        # # 即,在加入新元素后,将窗口中所有小于该值的数弹出;
        # # 将索引不在窗口内的数弹出,但这一操作依赖于索引,索引在维护窗口的时候,存的是数的索引
        # rem = []
        # res = []
        # for i, num in enumerate(nums):
        #     j = 0
        #     while j < len(rem):
        #         if nums[rem[j]] < num or rem[j] <= i - k:
        #             rem.pop(j)
        #         else:
        #             j += 1
        #     rem.append(i)
        #     if i >= k - 1:
        #         res.append(nums[rem[0]])
        # return res

        # 动态规划
        if not nums: return []
        n = len(nums)
        left_max = [0] * n
        left_max[0] = nums[0]
        right_max = [0] * n
        right_max[-1] = nums[-1]
        res = []
        for i in range(1, n):
            left_max[i] = nums[i] if i % k == 0 else max(left_max[i - 1], nums[i])
        for i in range(n - 2, -1, -1):
            right_max[i] = nums[i] if i % k == 0 else max(right_max[i + 1], nums[i])
        i = 0
        while i + k - 1 < n:
            res.append(max(right_max[i], left_max[i + k - 1]))
            i += 1
        return res


solve = Solution()
print(solve.maxSlidingWindow(nums, k))

# ############################## 240. 搜索二维矩阵 II ################################
# matrix = [
#   [1,   4,  7, 11, 15],
#   [2,   5,  8, 12, 19],
#   [3,   6,  9, 16, 22],
#   [10, 13, 14, 17, 24],
#   [18, 21, 23, 26, 30]]
# target = 5          # T
# target = 20         # F
#
# matrix = [[20]]
# target = 5          # T
# target = 20         # F
#
#
# class Solution:
#     def search_lis(self, lis, target):
#         n = len(lis)
#         i = 0
#         j = n - 1
#         while i <= j:
#             mid = (j + i) // 2
#             if lis[mid] == target:
#                 return True
#             elif lis[mid] > target:
#                 j = mid - 1
#             else:
#                 i = mid + 1
#         return False
#
#     def searchMatrix(self, matrix, target):
#         """
#         :type matrix: List[List[int]]
#         :type target: int
#         :rtype: bool
#         """
#         if not matrix or not matrix[0]:
#             return False
#         for lis in matrix:
#             if lis[0] <= target <= lis[-1]:
#                 if self.search_lis(lis, target):
#                     return True
#             elif target < lis[0]:
#                 break
#         return False
#
#
# class Solution1:
#     # O(m+n)的修建法,从左下角位置出发
#     # 如果当前元素大于target->上移
#     # 如果当前元素小于target->右移
#     # 直到找到元素或者超出索引
#     def searchMatrix(self, matrix, target):
#         # an empty matrix obviously does not contain `target` (make this check
#         # because we want to cache `width` for efficiency's sake)
#         if len(matrix) == 0 or len(matrix[0]) == 0:
#             return False
#
#         # cache these, as they won't change.
#         height = len(matrix)
#         width = len(matrix[0])
#
#         # start our "pointer" in the bottom-left
#         row = height - 1
#         col = 0
#
#         while col < width and row >= 0:
#             if matrix[row][col] > target:
#                 row -= 1
#             elif matrix[row][col] < target:
#                 col += 1
#             else:  # found it
#                 return True
#
#         return False
#
#
# solve = Solution()
# print(solve.searchMatrix(matrix, target))
