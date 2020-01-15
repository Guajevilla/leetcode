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

# ############################### 331. 验证二叉树的前序序列化 ################################
# x = "9,3,4,#,#,1,#,#,2,#,6,#,#"         # T
# # x = "1,#"                               # F
# # x = "9,#,#,1"                           # F
# # x = "1"                               # F
# # x = "#"                               # T
# # x = ""                               # F
# # x = "7,2,#,2,#,#,#,6,#"                 # F
#
#
# class Solution:
#     def isValidSerialization(self, preorder: str) -> bool:
#         # 用栈模拟前序遍历递归建树的过程
#         preorder = preorder.split(",")
#         stack = []
#         for item in preorder:
#             while stack and stack[-1] == "#" and item == "#":
#                 stack.pop()
#                 if not stack:return False
#                 stack.pop()
#             stack.append(item)
#         return stack == ["#"]
#         # return len(stack) == 1 and stack[0] == "#"
#
#         # # 通过计算接下来应有的叶子节点数量判断
#         # # 值得注意的是cnt不能在列表还没遍历完的时候就等于0，因为这意味着接下来不应该有元素了
#         # cnt = 1
#         # lis = preorder.split(",")
#         # for ele in lis:
#         #     if cnt == 0:
#         #         return False
#         #     if ele == '#':
#         #         cnt -= 1
#         #     else:
#         #         cnt += 1
#         #     if cnt < 0:
#         #         return False
#         # return cnt == 0
#         #
#         # # # 上个解法等价于下面这个，下面这个写的更简洁
#         # # preorder = preorder.split(",")
#         # # edges = 1
#         # # for item in preorder:
#         # #     edges -= 1
#         # #     if edges < 0: return False
#         # #     if item != "#":
#         # #         edges += 2
#         # # return edges == 0
#
#
# solve = Solution()
# print(solve.isValidSerialization(x))

# ############################### 332. 重新安排行程 ################################
x = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
# ["JFK", "MUC", "LHR", "SFO", "SJC"]
x = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
# ["JFK","ATL","JFK","SFO","ATL","SFO"]
x = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]
# ["JFK","NRT","JFK","KUL"]
x = [["JFK","AAA"],["AAA","JFK"],["JFK","BBB"],["JFK","CCC"],["CCC","JFK"]]
# ["JFK","AAA","JFK","CCC","JFK","BBB"]


class Solution:
    def findItinerary(self, tickets):
        # dic = {}
        # for ticket in tickets:
        #     if ticket[0] in dic:
        #         dic[ticket[0]].append(ticket[1])
        #     else:
        #         dic[ticket[0]] = [ticket[1]]
        # print(dic)
        # for ele in dic:
        #     if len(dic[ele]) > 1:
        #         dic[ele].sort()
        # print(dic)
        # ans = []
        # while dic:
        #     tmp = 'JFK'
        #     res = [tmp]
        #     while tmp in dic:
        #         if len(dic[tmp]) == 1:
        #             ttmp = dic[tmp][0]
        #             del dic[tmp]
        #             tmp = ttmp
        #         else:
        #             tmp = dic[tmp].pop(0)
        #         res.append(tmp)
        #     ans = res + ans[1:]
        # return ans


        from collections import defaultdict
        graph = defaultdict(list)
        res = []

        for x, y in sorted(tickets):
            graph[x].append(y)

        def dfs(f):  # 深搜函数
            while graph[f]:
                dfs(graph[f].pop(0))  # 路径检索
            res.insert(0, f)  # 放在最前

        dfs('JFK')
        return res


solve = Solution()
print(solve.findItinerary(x))

# ############################### 334. 递增的三元子序列 ################################
# # 给定一个未排序的数组，判断这个数组中是否存在长度为 3 的递增子序列。
# # 要求算法的时间复杂度为 O(n)，空间复杂度为 O(1) 。
# x = [1,2,3,4,5]         # true
# x = [5,4,3,2,1]         # false
# x = [1,5,4,2,3]         # true
# x = [5,1,4,2,3]         # true
# x = [2,5,3,4,7]         # true
#
#
# class Solution:
#     def increasingTriplet(self, nums) -> bool:
#         min_num, sec_num = float('inf'), float('inf')  # 初始化最小值，次小值
#
#         for num in nums:
#             if num <= min_num:
#                 min_num = num
#             elif num <= sec_num:
#                 sec_num = num
#             else:               # 找到了第三大的数
#                 return True
#
#         return False
#
#
# solve = Solution()
# print(solve.increasingTriplet(x))

# ############################### 335. 路径交叉 ################################
x = [2,1,1,2]           # true
x = [1,2,3,4]           # false
x = [1,1,1,1]           # true


class Solution:
    def isSelfCrossing(self, x) -> bool:
        if len(x) < 4: return False
        a, b, c, (d, e, f) = 0, 0, 0, x[:3]
        for i in range(3, len(x)):
            a, b, c, d, e, f = b, c, d, e, f, x[i]
            if e < c - a and f >= d: return True
            if c - a <= e <= c and f >= (d if d - b < 0 else d - b): return True
        return False


solve = Solution()
print(solve.isSelfCrossing(x))

# ############################### 336. 回文对 ################################
# x = ["abcd","dcba","lls","s","sssll"]           # [[0,1],[1,0],[3,2],[2,4]]
# x = ["bat","tab","cat"]                         # [[0,1],[1,0]]
#
#
# class Solution:
#     def palindromePairs(self, words):
#
#
# solve = Solution()
# print(solve.palindromePairs(x))

# ############################### 337. 打家劫舍 III ################################
x = stringToTreeNode('[3,2,3,null,3,null,1]')           # 7
x = stringToTreeNode('[3,4,5,1,3,null,1]')              # 9


class Solution:
    def rob(self, root: TreeNode) -> int:


solve = Solution()
print(solve.rob(x))
