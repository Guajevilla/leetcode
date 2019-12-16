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


# ############################### 257. 二叉树的所有路径 #################################
# x = stringToTreeNode('[1,2,3,null,5]')  # ["1->2->5", "1->3"]
# x = stringToTreeNode('[1]')  # ["1->2->5", "1->3"]
#
#
# class Solution:
#     def binaryTreePaths(self, root: TreeNode):
#         if not root:
#             return []
#         res = []
#
#         def dfs(root, s):
#             if not s:
#                 s = str(root.val)
#             else:
#                 s += '->' + str(root.val)
#             if not (root.left or root.right):
#                 res.append(s)
#                 return
#             if root.left:
#                 dfs(root.left, s)
#             if root.right:
#                 dfs(root.right, s)
#
#         dfs(root, '')
#         return res
#
#         # def construct_paths(root, path):
#         #     if root:
#         #         path += str(root.val)
#         #         if not root.left and not root.right:  # 当前节点是叶子节点
#         #             paths.append(path)  # 把路径加入到答案中
#         #         else:
#         #             path += '->'  # 当前节点不是叶子节点，继续递归遍历
#         #             construct_paths(root.left, path)
#         #             construct_paths(root.right, path)
#         #
#         # paths = []
#         # construct_paths(root, '')
#         # return paths
#
#         # 迭代
#         # if not root:
#         #     return []
#         #
#         # paths = []
#         # stack = [(root, str(root.val))]
#         # while stack:
#         #     node, path = stack.pop()
#         #     if not node.left and not node.right:
#         #         paths.append(path)
#         #     if node.left:
#         #         stack.append((node.left, path + '->' + str(node.left.val)))
#         #     if node.right:
#         #         stack.append((node.right, path + '->' + str(node.right.val)))
#         #
#         # return paths
#
#
# solve = Solution()
# prettyPrintTree(x)
# print(solve.binaryTreePaths(x))

# ############################### 258. 各位相加 #################################
# # 你可以不使用循环或者递归，且在 O(1) 时间复杂度内解决这个问题吗？
# x = 38      # 2
#
#
# class Solution:
#     def addDigits(self, num: int) -> int:
#         # 9的倍数各位和为9,其他递推..
#         if num == 0:
#             return 0
#         tmp = num % 9
#         if tmp == 0:
#             return 9
#         else:
#             return tmp
#
#         # if num == 0:
#         #     return 0
#         # return (num - 1) % 9 + 1
#
#
# solve = Solution()
# print(solve.addDigits(x))

# ############################### 260. 只出现一次的数字 III #################################
# 给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。
# 你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？
x = [1,2,1,3,2,5]       # [3,5]


class Solution:
    def singleNumber(self, nums):
        # # 用哈希,但是空间复杂度不满足常数,最坏情况需要 n//2+1
        # res = set()
        # for num in nums:
        #     if num not in res:
        #         res.add(num)
        #     else:
        #         res.remove(num)
        # return list(res)

        # 异或,其他元素都出现两次,两个需要的结果只出现一次,所有结果异或后的结果 = res[0] ^ res[1]
        # mask为找一位,在该位xor=1,这样可以把整个数组分为两组
        # 一组与mask与操作等于1,一组等于0
        # 这两组中必然一组有一个只出现一次的数
        xor = 0
        for num in nums:
            xor = xor ^ num
        mask = xor & (-xor)
        res = [0, 0]
        for num in nums:
            if (num & mask) == 0:
                res[0] ^= num
            else:
                res[1] ^= num
        return res


solve = Solution()
print(solve.singleNumber(x))
