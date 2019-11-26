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
# x = stringToTreeNode('[1,2,null,null,3]')
# # x = stringToTreeNode('[3,1,2]')
#
#
# class Solution:
#     def preorderTraversal(self, root: TreeNode):
#         # 迭代
#         if not root:
#             return []
#         stack = [root]
#         res = []
#         while stack:
#             tmp = stack.pop()
#             res.append(tmp.val)
#             if tmp.right:
#                 stack.append(tmp.right)
#             if tmp.left:
#                 stack.append(tmp.left)
#
#         return res
#
#         # # 递归
#         # res = []
#         #
#         # def pre_order(root):
#         #     if not root:
#         #         return
#         #     res.append(root.val)
#         #     pre_order(root.left)
#         #     pre_order(root.right)
#         #
#         # pre_order(root)
#         # return res
#
#
# solve = Solution()
# print(solve.preorderTraversal(x))

# ############################### 145. 二叉树的后序遍历 ###############################
x = stringToTreeNode('[1,2,null,null,3]')            # [3,2,1]
# x = stringToTreeNode('[1,null,2,3]')                # [3,2,1]
x = stringToTreeNode('[1,2,3,4,5,6,7]')            # [4,5,2,6,7,3,1]
# x = stringToTreeNode('[]')            # [3,2,1]


# class Solution:
#     def postorderTraversal(self, root: TreeNode):
#         # 迭代,当节点访问过才弹出
#         # if not root:
#         #     return []
#         # stack = [root]
#         # res = []
#         # rem = set()
#         # while stack:
#         #     tmp = stack[-1]
#         #     if tmp in rem or not (tmp.left or tmp.right):
#         #         res.append(tmp.val)
#         #         stack.pop()
#         #         continue
#         #     if tmp.right:
#         #         stack.append(tmp.right)
#         #     if tmp.left:
#         #         stack.append(tmp.left)
#         #     rem.add(tmp)
#         #
#         # return res
#
#
#         # 别人的迭代,利用前序反过来,模仿先序生成“根右左”再反转输出就是“左右根
#         res = []
#         p = root
#         stack = []
#         while p or stack:
#             while p:
#                 res.append(p.val)
#                 stack.append(p)
#                 p = p.right
#             p = stack.pop().left
#         return res[::-1]
#
#
#         # # 递归
#         # res = []
#         #
#         # def post_order(root):
#         #     if not root:
#         #         return
#         #     post_order(root.left)
#         #     post_order(root.right)
#         #     res.append(root.val)
#         #
#         # post_order(root)
#         # return res
#
#
# solve = Solution()
# print(solve.postorderTraversal(x))

# ############################### 146. LRU缓存机制 ###############################
# 获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
# 写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。
# O(1) 时间复杂度

# from collections import OrderedDict
#
#
# # 主要用到了OrderedDict有序字典
# # 主要函数:move_to_end(key)将键值对移到队尾
# # popitem(last=False)弹出队首键值对
# # popitem弹出队尾键值对
# # 利用popitem(last=False)[0]可取弹出的键
# # 利用popitem(last=False)[1]可取弹出的值
# class LRUCache(object):
#     def __init__(self, capacity):
#         """
#         :type capacity: int
#         """
#         self.capacity = capacity
#         self.cache = OrderedDict()
#
#     def get(self, key):
#         """
#         :type key: int
#         :rtype: int
#         """
#         if key in self.cache:
#             self.cache.move_to_end(key)
#             return self.cache[key]
#         else:
#             return -1
#
#     def put(self, key, value):
#         """
#         :type key: int
#         :type value: int
#         :rtype: None
#         """
#         if key not in self.cache:
#             if len(self.cache) == self.capacity:
#                 self.cache.popitem(last=False)
#         else:
#             del self.cache[key]
#         self.cache[key] = value


# # 双向链表解决,主要想法同上,键对应的值是一个node
# class Node:
#     def __init__(self, key, val):
#         self.key = key
#         self.val = val
#         self.prev = None
#         self.next = None
#
#
# class LRUCache:
#
#     def __init__(self, capacity: int):
#         # 构建首尾节点, 使之相连
#         self.head = Node(0, 0)
#         self.tail = Node(0, 0)
#         self.head.next = self.tail
#         self.tail.prev = self.head
#
#         self.lookup = dict()
#         self.max_len = capacity
#
#     def get(self, key: int) -> int:
#         if key in self.lookup:
#             node = self.lookup[key]
#             self.remove(node)
#             self.add(node)
#             return node.val
#         else:
#             return -1
#
#     def put(self, key: int, value: int) -> None:
#         if key in self.lookup:
#             self.remove(self.lookup[key])
#         if len(self.lookup) == self.max_len:
#             # 把表头位置节点删除(说明最近的数据值)
#             self.remove(self.head.next)
#         self.add(Node(key, value))
#
#     # 删除链表节点
#     def remove(self, node):
#         del self.lookup[node.key]
#         node.prev.next = node.next
#         node.next.prev = node.prev
#
#     # 加在链表尾
#     def add(self, node):
#         self.lookup[node.key] = node
#         pre_tail = self.tail.prev
#         node.next = self.tail
#         self.tail.prev = node
#         pre_tail.next = node
#         node.prev = pre_tail
#
#
# # Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(2)
# print(obj.get(1))
# obj.put(1,1)
# obj.put(2,2)
# print(obj.get(1))
# obj.put(3,3)
# print(obj.get(2))
# obj.put(4,4)
# print(obj.get(1))
# print(obj.get(3))
# print(obj.get(4))


# # ############################### 147. 对链表进行插入排序 ###############################
# x = stringToListNode('[4,2,1,3]')
# x = stringToListNode('[3,5,6,1,8,7,2,4]')
# x = stringToListNode('[1,5,3,4,0]')
# # x = stringToListNode('[1]')
#
#
# class Solution(object):
#     def insertionSortList(self, head):
#         """
#         :type head: ListNode
#         :rtype: ListNode
#         """
#         # O(N^2)
#         if not (head and head.next):
#             return head
#         dummy = ListNode(0)
#         dummy.next = head
#         p = head.next
#         pre = head
#         while p:
#             if p.val >= pre.val:
#                 p = p.next
#                 pre = pre.next
#                 continue
#             else:
#                 pre.next = p.next
#                 tmp = dummy
#                 while tmp.next and p.val > tmp.next.val:
#                     tmp = tmp.next
#                 p.next = tmp.next
#                 tmp.next = p
#                 p = pre.next
#         return dummy.next
#
#
# solve = Solution()
# printList(solve.insertionSortList(x))

# # ############################### 148. 排序链表 ###############################
# O(n log n) 时间复杂度和常数级空间复杂度


class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """


solve = Solution()
printList(solve.sortList(x))

# ############################### 149. 直线上最多的点数 ###############################
# x = [[1,1],[2,2],[3,3]]     # 3
# x = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]   # 4
# x = [[1,1],[0,0],[1,1]]   # 3
# x = [[1,1],[0,0]]   # 2
# x = [[1,1],[1,1]]   # 2
# x = [[1,1],[0,0],[1,1],[0,0],[2,3]]   # 4
# x = [[1,1],[1,1],[1,1],[0,1],[10,11]]   # 4
# x = [[1,1],[1,1],[1,1],[0,1]]   # 4
# x = [[2,3],[3,3],[-5,3],[2,1]]   # 3
# x = [[3,-1],[3,2],[3,1],[2,1]]   # 3
# x = [[84,250],[0,0],[1,0],[0,-70],[0,-70],[1,-1],[21,10],[42,90],[-42,-230]]   # 6
# x = [[94911152, 94911151],[0,0],[94911151, 94911150]]   # 2
# x = [[0,-12],[5,2],[2,5],[0,-5],[1,5],[2,-2],[5,-4],[3,4],[-2,4],[-1,4],[0,-5],[0,-8],[-2,-1],[0,-11],[0,-9]]
# x = [[3,10],[0,2],[0,2],[3,10]]     # 4
# x = [[-435,-347],[-435,-347],[609,613],[-348,-267],[-174,-107],[87,133],[-87,-27],[-609,-507],[435,453],[-870,-747],[-783,-667],[0,53],[-174,-107],[783,773],[-261,-187],[-609,-507],[-261,-187],[-87,-27],[87,133],[783,773],[-783,-667],[-609,-507],[-435,-347],[783,773],[-870,-747],[87,133],[87,133],[870,853],[696,693],[0,53],[174,213],[-783,-667],[-609,-507],[261,293],[435,453],[261,293],[435,453]]
# # 37
#
#
# class Solution(object):
#     def maxPoints(self, points):
#         """
#         :type points: List[List[int]]
#         :rtype: int
#         """
#         if not points:
#             return 0
#         cnt = 1
#         for i, point1 in enumerate(points):
#             repeat = 0
#             for j, point2 in enumerate(points[i+1:]):
#                 tmp = 1
#                 if point2[0] == point1[0]:
#                     if point2 == point1:
#                         repeat += 1
#                         cnt = max(cnt, tmp + repeat)
#                         continue
#                     else:
#                         tmp += 1
#                         for point in points[i+j+2:]:
#                             if point[0] == point1[0]:
#                                 tmp += 1
#                         cnt = max(cnt, tmp + repeat)
#                         break
#                 else:
#                     slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
#                     tmp += 1
#                     for point in points[i+j+2:]:
#                         if point[0] == point1[0]:
#                             if point == point1 and j == 0:
#                                 repeat += 1
#                             continue
#                         if (point[1] - point1[1]) / (point[0] - point1[0]) == slope:
#                             tmp += 1
#                 cnt = max(cnt, tmp + repeat)
#         return cnt
#
#
#
#         from collections import Counter, defaultdict
#         # 所有点统计
#         points_dict = Counter(tuple(point) for point in points)
#         # 把唯一点列举出来
#         not_repeat_points = list(points_dict.keys())
#         n = len(not_repeat_points)
#         if n == 1: return points_dict[not_repeat_points[0]]
#         res = 0
#
#         # 求最大公约数
#         def gcd(x, y):
#             if y == 0:
#                 return x
#             else:
#                 return gcd(y, x % y)
#
#         for i in range(n - 1):
#             # 点1
#             x1, y1 = not_repeat_points[i][0], not_repeat_points[i][1]
#             # 斜率
#             slope = defaultdict(int)
#             for j in range(i + 1, n):
#                 # 点2
#                 x2, y2 = not_repeat_points[j][0], not_repeat_points[j][1]
#                 dy, dx = y2 - y1, x2 - x1
#                 # 方式一 利用公约数
#                 g = gcd(dy, dx)
#                 if g != 0:
#                     dy //= g
#                     dx //= g
#                 slope["{}/{}".format(dy, dx)] += points_dict[not_repeat_points[j]]
#                 # --------------------
#                 # 方式二, 利用除法(不准确, 速度快)
#                 # if dx == 0:
#                 #     tmp = "#"
#                 # else:
#                 #     tmp = dy * 1000 / dx * 1000
#                 # slope[tmp] += points_dict[not_repeat_points[j]]
#                 # ------------------------------
#             res = max(res, max(slope.values()) + points_dict[not_repeat_points[i]])
#         return res
#
#
# solve = Solution()
# print(solve.maxPoints(x))

# ############################### 150. 逆波兰表达式求值 ###############################
# x = ["2", "1", "+", "3", "*"]       # 9
# x = ["4", "13", "5", "/", "+"]      # 6
# x = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]   # 22
# x = ["-78","-33","196","+","-19","-","115","+","-","-99","/","-18","8","*","-86","-","-","16","/","26","-14","-","-","47","-","101","-","163","*","143","-","0","-","171","+","120","*","-60","+","156","/","173","/","-24","11","+","21","/","*","44","*","180","70","-40","-","*","86","132","-84","+","*","-","38","/","/","21","28","/","+","83","/","-31","156","-","+","28","/","95","-","120","+","8","*","90","-","-94","*","-73","/","-62","/","93","*","196","-","-59","+","187","-","143","/","-79","-89","+","-"]
# # 165
#
#
# class Solution(object):
#     def evalRPN(self, tokens):
#         """
#         :type tokens: List[str]
#         :rtype: int
#         """
#         num = []
#         ops = {'+', '-', '*', '/'}
#         for ele in tokens:
#             if ele in ops:
#                 num0 = num.pop()
#                 num1 = num.pop()
#                 if ele == '+':
#                     num.append(num1 + num0)
#                 elif ele == '-':
#                     num.append(num1 - num0)
#                 elif ele == '*':
#                     num.append(num1 * num0)
#                 else:
#                     # num.append(num1 // num0)
#                     num.append(int(num1 / float(num0)))
#             else:
#                 num.append(int(ele))
#
#         return num[0]
#
#
# solve = Solution()
# print(solve.evalRPN(x))
