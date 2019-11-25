import json
import time
# ############################### 141. 环形链表 I ###############################


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


head = stringToListNode('[1,2,3]')


# class Solution:
#     def hasCycle(self, head: ListNode) -> bool:
#         # # 用哈希表存经过的节点,用set时间复杂度为O(1)
#         # lookup = set()
#         # p = head
#         # while p:
#         #     lookup.add(p)
#         #     if p.next in lookup:
#         #         return True
#         #     p = p.next
#         # return False
#
#         # 快慢指针法,每次多走一步,快的总会追上慢的
#         slow = head
#         fast = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next
#             if slow == fast:
#                 return True
#         return False
#
#
# solve = Solution()
# print(solve.hasCycle(head))

# ############################### 142. 环形链表 II ###############################


# class Solution:
#     def detectCycle(self, head: ListNode) -> ListNode:
#         # # 哈希
#         # lookup = set()
#         # p = head
#         # cnt = 0
#         # while p:
#         #     lookup.add(p)
#         #     cnt += 1
#         #     if p.next in lookup:
#         #         return p.next
#         #     p = p.next
#         # return None
#
#         # 快慢指针,到相遇点后,再分别从相遇点和起点出发.再次相遇点即为入口节点
#         slow = head
#         fast = head
#         start = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next
#             if slow == fast:
#                 while slow != start:
#                     slow = slow.next
#                     start = start.next
#                 return slow
#         return None

# ############################### 143. 重排链表 ###############################
# # 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
# x = stringToListNode('[1,2,3,4]')       # [1,4,2,3]
# x = stringToListNode('[1,2,3,4,5]')       # [1,5,2,4,3]
#
#
# class Solution:
#     def reorderList(self, head: ListNode) -> None:
#         """
#         Do not return anything, modify head in-place instead.
#         """
#         # 用栈记住之前的节点
#         # if not head or not head.next:
#         #     return head
#         # rem = []
#         # p = head.next
#         # while p:
#         #     rem.append(p)
#         #     p = p.next
#         #
#         # p = head
#         # flag = 0
#         # while rem:
#         #     if flag:
#         #         p.next = rem.pop(0)
#         #         flag = 0
#         #     else:
#         #         p.next = rem.pop()
#         #         flag = 1
#         #     p = p.next
#         # p.next = None
#
#         # # 找到链表的中点,把后面那个反转后与前半部分依次插入
#         if not head or not head.next: return head
#         fast = head
#         pre_mid = head
#         # 找到中点, 偶数个找到时上界那个
#         while fast.next and fast.next.next:
#             pre_mid = pre_mid.next
#             fast = fast.next.next
#         # 翻转中点之后的链表,采用是pre, cur双指针方法
#         pre = None
#         cur = pre_mid.next
#         # 1 2 5 4 3
#         while cur:
#             tmp = cur.next
#             cur.next = pre
#             pre = cur
#             cur = tmp
#         # 翻转链表和前面链表拼接
#         pre_mid.next = pre
#         # 1 5 2 4 3
#         # 链表头
#         p1 = head
#         # 翻转头
#         p2 = pre_mid.next
#         # print(p1.val, p2.val)
#         while p1 != pre_mid:
#             pre_mid.next = p2.next
#             p2.next = p1.next
#             p1.next = p2
#             p1 = p2.next
#             p2 = pre_mid.next
#
#
# solve = Solution()
# solve.reorderList(x)
# printList(x)

# ############################### 144. 二叉树的前序遍历 ###############################
x = stringToTreeNode('[1,2,null,null,3]')
# x = stringToTreeNode('[3,1,2]')


class Solution:
    def preorderTraversal(self, root: TreeNode):
        # 迭代
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            tmp = stack.pop()
            res.append(tmp.val)
            if tmp.right:
                stack.append(tmp.right)
            if tmp.left:
                stack.append(tmp.left)

        return res

        # # 递归
        # res = []
        #
        # def pre_order(root):
        #     if not root:
        #         return
        #     res.append(root.val)
        #     pre_order(root.left)
        #     pre_order(root.right)
        #
        # pre_order(root)
        # return res


solve = Solution()
print(solve.preorderTraversal(x))

# ############################### 145. 二叉树的后序遍历 ###############################
x = stringToTreeNode('[1,2,null,null,3]')            # [3,2,1]
# x = stringToTreeNode('[1,null,2,3]')                # [3,2,1]
x = stringToTreeNode('[1,2,3,4,5,6,7]')            # [4,5,2,6,7,3,1]
# x = stringToTreeNode('[]')            # [3,2,1]


class Solution:
    def postorderTraversal(self, root: TreeNode):
        # 迭代,当节点访问过才弹出
        # if not root:
        #     return []
        # stack = [root]
        # res = []
        # rem = set()
        # while stack:
        #     tmp = stack[-1]
        #     if tmp in rem or not (tmp.left or tmp.right):
        #         res.append(tmp.val)
        #         stack.pop()
        #         continue
        #     if tmp.right:
        #         stack.append(tmp.right)
        #     if tmp.left:
        #         stack.append(tmp.left)
        #     rem.add(tmp)
        #
        # return res


        # 别人的迭代,利用前序反过来
        res = []
        p = root
        stack = []
        while p or stack:
            while p:
                res.append(p.val)
                stack.append(p)
                p = p.right
            p = stack.pop().left
        return res[::-1]


        # # 递归
        # res = []
        #
        # def post_order(root):
        #     if not root:
        #         return
        #     post_order(root.left)
        #     post_order(root.right)
        #     res.append(root.val)
        #
        # post_order(root)
        # return res


solve = Solution()
print(solve.postorderTraversal(x))
