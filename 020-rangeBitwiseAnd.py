import math
import json
# ############################### 201. 数字范围按位与 ###############################
m = 5
n = 7       #　4

# m = 0
# n = 1       #　0
#
# m = 5
# n = 8       #　0

# m = 1
# n = 2147483646       # 0

# m = 5
# n = 5       #　4

# m = int('1101', 2)
# n = int('1111', 2)      # 12


# class Solution:
#     def rangeBitwiseAnd(self, m: int, n: int) -> int:
#         # 这题就是找某一位未出现过0, 这样的位上置1,其他为0,的数大小
#         # 我的想法是从小的那个数出发,如果第i位是1,并且想保证这一位未出现过0的话,需要满足
#         # 比该数大,且这一位为0的数不在n范围内
#         # 这个数怎么找?其实就是m右移len-i-1位,+1,再左移len-i-1位回来
#         # 如:1010的左数第2个1,要保证这一位未出现过0,则1100不在范围内即可.
#         str_m = bin(m)[2:]
#         length = len(str_m)
#         res = 0
#         for i in range(length):
#             if str_m[i] == '1':
#                 if n < (((m >> (length - i - 1)) + 1) << (length - i - 1)):
#                     res += (1 << (length - i - 1))
#
#         return res
#
#         # # n的二进制位比m二进制最左边的1高时， & 的结果必然为0；
#         # # 由这个思想启发，二进制最高位相同时，这个1会保存，然后比较右一位，相同也保留...
#         # # 所以只需要 m n 同时右移到相等时
#         # # m n的值就是 & 后能保留的位数，然后左移回来就是最后的值。
#         # i = 0
#         # while m != n:
#         #     m >>= 1
#         #     n >>= 1
#         #     i += 1
#         # return m << i
#
#
# solve = Solution()
# print(solve.rangeBitwiseAnd(m, n))

# ############################### 202. 快乐数 ###############################
# x = 19      # T
#
#
# class Solution:
#     def isHappy(self, n: int) -> bool:
#         # # 略傻的办法..我把20以内的非快乐数算出来了,发现非快乐数都是循环的,把这些非快乐树存起来...
#         # if n <= 0:
#         #     return False
#         # rem = {2,3,4,5,6,8,9,11,12,14,15,16,17,18}
#         # while n not in rem and n != 1:
#         #     s_tmp = str(n)
#         #     n = 0
#         #     for i in s_tmp:
#         #         n += int(i) ** 2
#         # if n == 1:
#         #     return True
#         # else:
#         #     return False
#
#         # # 还可以记录过程中的数,有重复说明循环了
#         # n = str(n)
#         # visited = set()
#         # while 1:
#         #     n = str(sum(int(i) ** 2 for i in n))
#         #     if n == "1":
#         #         return True
#         #     if n in visited:
#         #         return False
#         #     visited.add(n)
#
#         # 快慢指针,由于每次快一步,必然会有相等的时候
#         n = str(n)
#         slow = n
#         fast = str(sum(int(i) ** 2 for i in n))
#         while slow != fast:
#             slow = str(sum(int(i) ** 2 for i in slow))
#             fast = str(sum(int(i) ** 2 for i in fast))
#             fast = str(sum(int(i) ** 2 for i in fast))
#         return slow == "1"
#
#
# solve = Solution()
# for i in range(25):
#     if solve.isHappy(i):
#         print(i)

# ############################### 203. 移除链表元素 ###############################


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


x = stringToListNode('[1,2,6,3,4,5,6]')
val = 6

x = stringToListNode('[6,1]')
val = 6


# class Solution:
#     def removeElements(self, head: ListNode, val: int) -> ListNode:
#         # dummy = ListNode(None)
#         # dummy.next = head
#         # pre = dummy
#         # while head:
#         #     if head.val == val:
#         #         pre.next = head.next
#         #         head = head.next
#         #     else:
#         #         pre = pre.next
#         #         head = head.next
#         # return dummy.next
#
#         # 递归
#         if not head:
#             return
#         head.next = self.removeElements(head.next, val)
#         return head.next if head.val == val else head
#
#
# solve = Solution()
# printList(solve.removeElements(x, val))

# ############################### 204. 计数质数 ###############################
# # 统计所有小于非负整数 n 的质数的数量。
# x = 10      # 4
#
#
# class Solution:
#     def countPrimes(self, n: int) -> int:
#         # cnt = 0
#         # if n < 2:
#         #     return cnt
#         # dp = [1] * n
#         # for i in range(2, n):
#         #     if dp[i] == 1:
#         #         cnt += 1
#         #         j = i
#         #         while i * j < n:
#         #             dp[i * j] = 0
#         #             j += 1
#         # return cnt
#
#         # 这个算法的加速点在于,上面在去除非质数的过程中有重复,当i大于 n ** 0.5 后,即为重复计算了
#         # 因为这个时候相当于从原来的 small*large 变成了 large*small
#         if n < 2: return 0
#         isPrimes = [1] * n
#         isPrimes[0] = isPrimes[1] = 0
#         for i in range(2, int(n ** 0.5) + 1):
#             if isPrimes[i] == 1:
#                 isPrimes[i * i: n: i] = [0] * len(isPrimes[i * i: n: i])
#         return sum(isPrimes)
#
#
# solve = Solution()
# print(solve.countPrimes(x))

# ############################### 205. 同构字符串 ###############################
s = "egg"
t = "add"       # T

# s = "ab"
# t = "aa"       # F

# s = "ab"
# t = "ba"       # T
#
# s = "foo"
# t = "bar"       # F
#
# s = "paper"
# t = "title"     # T


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        dic = {}
        for i in range(len(s)):
            if s[i] not in dic:
                if t[i] not in dic.values():
                    dic[s[i]] = t[i]
                else:
                    return False
            else:
                if dic[s[i]] != t[i]:
                    return False

        return True


solve = Solution()
print(solve.isIsomorphic(s, t))

# ############################### 206. 反转链表 ###############################
# 可用迭代或递归
x = stringToListNode('[1,2]')
# x = stringToListNode('[]')


class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # # 迭代
        # stack = []
        # dummy = ListNode(0)
        # while head:
        #     stack.append(head)
        #     head = head.next
        #
        # p = dummy
        # while stack:
        #     p.next = stack.pop()
        #     p = p.next
        # p.next = None
        # return dummy.next

        # # 递归
        # if not head:
        #     return head
        # res = [None]
        #
        # def sub_solver(head):
        #     if not head.next:
        #         res[0] = head
        #         return head
        #     tmp = sub_solver(head.next)
        #     tmp.next = head
        #     return head
        #
        # sub_solver(head).next = None
        # return res[0]

        # 别人的递归
        if not head or not head.next: return head
        new_head = self.reverseList(head.next)
        next_node = head.next
        next_node.next = head
        head.next = None
        return new_head

        # 别人的迭代
        prev = None
        cur = head
        while cur:
            # 保持下一个节点
            nxt = cur.next
            # 翻转
            cur.next = prev
            # 进行下一个
            prev = cur
            cur = nxt
        return prev


solve = Solution()
printList(solve.reverseList(x))
