# ############################### 191. 位1的个数 ###############################
# x = int('00000000000000000000000000001011', 2)      # 3
# x = int('00000000000000000000000010000000', 2)      # 1
# x = int('11111111111111111111111111111101', 2)      # 31
#
#
# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         # # s = str(bin(n))[2:]
#         # # bin函数后的结果就是str
#         # s = bin(n)[2:]
#         # cnt = 0
#         # for ele in s:
#         #     if ele == '1':
#         #         cnt += 1
#         # return cnt
#         #
#         # # 上述过程可以调用函数
#         # return bin(n).count('1')
#         #
#         # # 利用十进制转二进制,每次对2取余,为1则有一个1
#         # count = 0
#         # while n:
#         #     res = n % 2
#         #     if res == 1:
#         #         count += 1
#         #     n //= 2
#         # return count
#
#         # 位运算 把 n 与 1 进行与运算，将得到 n 的最低位数字
#         # 取出最低位数，再将 n 右移一位。循环此步骤，直到 n 等于零。
#         count = 0
#         while n:
#             count += n & 1
#             n >>= 1
#         return count
#
#
# solve = Solution()
# print(solve.hammingWeight(x))

# ############################### 198. 打家劫舍 ###############################
# x = [1,2,3,1]           # 4
# x = [2,7,9,3,1]         # 12
# x = [2,1,1,2]           # 4
#
#
# # 这题主要是最开始有个误解,以为必须隔一个取一个,其实只要隔开就行,不管隔开几个
# class Solution:
#     def rob(self, nums):
#         # if not nums:
#         #     return 0
#         # elif len(nums) == 1:
#         #     return nums[0]
#         # dp = [0] * len(nums)
#         # dp[0] = nums[0]
#         # dp[1] = max(nums[0], nums[1])
#         # for i in range(2, len(nums)):
#         #     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
#         # return dp[-1]
#
#         # 奇偶求和 其实和动态规划的思路一样,每一步都需要保存奇偶的最大值
#         odd = 0
#         even = 0
#         for i, num in enumerate(nums):
#             if i % 2:
#                 odd = max(num + odd, even)
#             else:
#                 even = max(num + even, odd)
#         return max(odd, even)
#
#
# solve = Solution()
# print(solve.rob(x))

# ############################### 199. 二叉树的右视图 ###############################


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


# 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
# x = stringToTreeNode('[1,2,3,null,5,null,4]')       # [1,3,4]
# x = stringToTreeNode('[1,2,null,3,4,5]')       # [1,3,4]
# x = stringToTreeNode('[1,2,3,4]')       # [1,3,4]
# prettyPrintTree(x)
#
#
# class Solution:
#     def rightSideView(self, root: TreeNode):
#         # # 层次遍历取每层最后一个
#         # res = []
#         #
#         # def sub_solver(root, depth):
#         #     if not root:
#         #         return
#         #     if len(res) <= depth:
#         #         res.append([])
#         #     sub_solver(root.left, depth + 1)
#         #     sub_solver(root.right, depth + 1)
#         #     res[depth].append(root.val)
#         #
#         # sub_solver(root, 0)
#         #
#         # ans = []
#         # for lis in res:
#         #     ans.append(lis[-1])
#         # return ans
#
#         # # 其实并不需要存每次的遍历
#         # # 先深度搜索右子树,这样,只要是第一次到该深度,就可以保证是答案之一
#         # def dfs(node, res, depth):
#         #     if node is None:
#         #         return
#         #     if len(res) == depth:
#         #         res.append(node.val)
#         #     dfs(node.right, res, depth + 1)
#         #     dfs(node.left, res, depth + 1)
#         #
#         # res = []
#         # dfs(root, res, 0)
#         # return res
#
#         # 用一个队列,每次存满一层的元素
#         if not root:
#             return []
#         queue = [root]
#         res = []
#         while queue:
#             cur_size = len(queue)
#             res.append(queue[-1].val)
#             # 这里要注意，上一层的结点要全部出列
#             for _ in range(cur_size):
#                 top = queue.pop(0)
#                 if top.left:
#                     queue.append(top.left)
#                 if top.right:
#                     queue.append(top.right)
#         return res
#
#
# solve = Solution()
# print(solve.rightSideView(x))

# ############################### 200. 岛屿数量 ###############################
x = [['1','1','0','0','0'],
     ['1','1','0','0','0'],
     ['0','0','1','0','0'],
     ['0','0','0','1','1']]     # 3

x = [['1','1','1','1','0'],
     ['1','1','0','1','0'],
     ['1','1','0','0','0'],
     ['0','0','0','0','0']]     # 1

x = [["0","1","0"],
     ["1","0","1"],
     ["0","1","0"]]

x = [["1","1","1"],
     ["1","0","1"],
     ["1","1","1"]]


class Solution:
    def numIslands(self, grid):
        # 广度优先搜索
        from collections import deque
        if not grid: return 0
        row = len(grid)
        col = len(grid[0])
        cnt = 0

        def bfs(i, j):
            queue = deque()
            queue.appendleft((i, j))
            grid[i][j] = "0"
            while queue:
                i, j = queue.pop()
                for x, y in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                    tmp_i = i + x
                    tmp_j = j + y
                    if 0 <= tmp_i < row and 0 <= tmp_j < col and grid[tmp_i][tmp_j] == "1":
                        grid[tmp_i][tmp_j] = "0"
                        queue.appendleft((tmp_i, tmp_j))

        for i in range(row):
            for j in range(col):
                if grid[i][j] == "1":
                    bfs(i, j)
                    cnt += 1
        return cnt

        # 深度优先搜索
        # if not grid: return 0
        # row = len(grid)
        # col = len(grid[0])
        # cnt = 0
        #
        # def dfs(i, j):
        #     grid[i][j] = "0"
        #     for x, y in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        #         tmp_i = i + x
        #         tmp_j = j + y
        #         if 0 <= tmp_i < row and 0 <= tmp_j < col and grid[tmp_i][tmp_j] == "1":
        #             dfs(tmp_i, tmp_j)
        #
        # for i in range(row):
        #     for j in range(col):
        #         if grid[i][j] == "1":
        #             dfs(i, j)
        #             cnt += 1
        # return cnt


solve = Solution()
print(solve.numIslands(x))
