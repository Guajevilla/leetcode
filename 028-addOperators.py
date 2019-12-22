# ############################### 282. 给表达式添加运算符 #################################
# +, - 或 *
num = "123"
target = 6              # ["1+2+3", "1*2*3"]

num = "232"
target = 8              # ["2*3+2", "2+3*2"]

num = "105"
target = 5              # ["1*0+5","10-5"]

num = "00"
target = 0              # ["0+0", "0-0", "0*0"]

num = "03"
target = 3              # ["0+0", "0-0", "0*0"]

num = "3456237490"
target = 9191           # []

num = "000"
target = 0           # []


class Solution:
    def addOperators(self, num: str, target: int):
        # 超时且有'000'错误
        res = []

        def backtrack(s, start, j):
            if j != -1 and j > 0 and s[j-1] == '0':
                return
            if j == -1 or (j != -1 and(j >= len(s) - 1 or not s[j+1].isdigit())):
                if eval(s) == target:
                    res.append(s)

            # if max(1, abs(eval(s[:start]))) * (int(s[start:])) < target:
            #     return

            for i in range(start, len(s)):
                if s[i] != '0':
                    for ele in ['*', '+', '-']:
                        backtrack(s[:i]+ele+s[i:], i+2, -1)
                else:
                    for ele in ['*', '+', '-']:
                        backtrack(s[:i]+ele+s[i:], i+2, i+1)

        tmp = -1
        if num[0] == '0':
            tmp = 0
        backtrack(num, 1, tmp)
        return res


        # res = []
        # # 参数列表， 位置， 之前输出， 之前综合， 前一个数
        # def helper(index, preOutStr, preSum, preValue):
        #     if index == len(num):
        #         if preSum == target:
        #             res.append(preOutStr)
        #         return
        #     # 后面的数都做乘法, 还小于前面总和
        #     if max(1, abs(preValue)) * (int(num[index:])) < abs(target - preSum):
        #         return
        #     for i in range(index, index + 1 if num[index] == '0' else len(num)):
        #         cur = num[index:i + 1]
        #         curValue = int(cur)
        #         if not preOutStr:
        #             helper(i + 1, cur, curValue, curValue)
        #         else:
        #             helper(i + 1, preOutStr + '+' + cur, preSum + curValue, curValue)
        #             helper(i + 1, preOutStr + '-' + cur, preSum - curValue, -curValue)
        #             helper(i + 1, preOutStr + '*' + cur, preSum - preValue + curValue * preValue, curValue * preValue)
        # helper(0, '', 0, 0)
        # return res


solve = Solution()
print(solve.addOperators(num, target))

# ############################### 283. 移动零 #################################
# nums = [0,1,0,3,12]         # [1,3,12,0,0]
# # nums = []
#
#
# class Solution:
#     def moveZeroes(self, nums) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         # 先把非0放到前面
#         loc = 0
#         for num in nums:
#             if num != 0:
#                 nums[loc] = num
#                 loc += 1
#         # 再把后面的全置为0
#         while loc < len(nums):
#             nums[loc] = 0
#             loc += 1
#
#         # nums.pop(i)时间复杂度其实是O(n)
#         # i = len(nums) - 1
#         # while i >= 0:
#         #     if nums[i] == 0:
#         #         nums.append(nums.pop(i))
#         #     i -= 1
#
#
# solve = Solution()
# solve.moveZeroes(nums)
# print(nums)

# ############################### 284. 顶端迭代器 #################################
# class PeekingIterator:
#     def __init__(self, iterator):
#         """
#         Initialize your data structure here.
#         :type iterator: Iterator
#         """
#         self.it = iterator
#         self.next_num = iterator.next()
#
#     def peek(self):
#         """
#         Returns the next element in the iteration without advancing the iterator.
#         :rtype: int
#         """
#         return self.next_num
#
#     def next(self):
#         """
#         :rtype: int
#         """
#         tmp = self.next_num
#         if self.it.hasNext():
#             self.next_num = self.it.next()
#         else:
#             self.next_num = None
#         return tmp
#
#     def hasNext(self):
#         """
#         :rtype: bool
#         """
#         if self.next_num:
#             return True
#         return False

# ############################### 287. 寻找重复数 #################################
# # 不能更改原数组（假设数组是只读的）。
# # 只能使用额外的 O(1) 的空间。
# # 时间复杂度小于 O(n^2) 。
# # 数组中只有一个重复的数字，但它可能不止重复出现一次。
# x = [1,3,4,2,2]         # 2
# # x = [3,1,3,4,2]         # 3
# # x = [3,1,3,3]         # 3
# # x = [3,2,1,1]         # 3
#
#
# class Solution:
#     def findDuplicate(self, nums) -> int:
#         # 快慢指针法,数组有重复元素时候, 通过索引号移动会有环出现
#         # 这样就变成求环的第一个节点
#         slow = nums[0]
#         fast = nums[nums[0]]
#         while slow != fast:
#             # print(slow, fast)
#             slow = nums[slow]
#             fast = nums[nums[fast]]
#         slow = 0
#         while slow != fast:
#             slow = nums[slow]
#             fast = nums[fast]
#         return slow
#
#
#         # # # 二分法
#         # # l = 0
#         # # r = len(nums) - 1
#         # # n = r
#         # # while r > l + 1:
#         # #     mid = (r + l) >> 1
#         # #     greater = 0
#         # #     for num in nums:
#         # #         if num > mid:
#         # #             greater += 1
#         # #     if greater > n - mid:
#         # #         l = mid
#         # #     else:
#         # #         r = mid
#         # # return r
#         #
#         # # 等价的二分法
#         # left = 1
#         # right = len(nums) - 1
#         #
#         # while left < right:
#         #     mid = left + (right - left) // 2
#         #     # 计数， 找小于等于mid的个数
#         #     cnt = 0
#         #     for num in nums:
#         #         if num <= mid:
#         #             cnt += 1
#         #     # 根据鸽巢原理, https://baike.baidu.com/item/抽屉原理/233776?fromtitle=鸽巢原理&fromid=731656&fr=aladdin
#         #     # <= 说明 重复元素再右半边
#         #     if cnt <= mid:
#         #         left = mid + 1
#         #     else:
#         #         right = mid
#         # return left
#
#
# solve = Solution()
# print(solve.findDuplicate(x))

# ############################### 289. 生命游戏 #################################
# x = [
#   [0,1,0],
#   [0,0,1],
#   [1,1,1],
#   [0,0,0]]
#
# # [[0,0,0],
# #  [1,0,1],
# #  [0,1,1],
# #  [0,1,0]]
#
#
# class Solution:
#     def gameOfLife(self, board) -> None:
#         """
#         Do not return anything, modify board in-place instead.
#         """
#         # 用-1表示原来是1,变成了0
#         # 用2表示原来是0,变成了1
#         def count_alive(i, j):
#             cnt = 0
#             for r, c in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
#                 x, y = i + r, j + c
#                 if 0 <= x < m and 0 <= y < n:
#                     if board[x][y] == 1 or board[x][y] == -1:
#                         cnt += 1
#             return cnt
#
#         if not board or not board[0]:
#             return
#         m = len(board)
#         n = len(board[0])
#         for i in range(m):
#             for j in range(n):
#                 if board[i][j] == 0:
#                     cnt = count_alive(i, j)
#                     if cnt == 3:
#                         board[i][j] = 2
#                 else:
#                     cnt = count_alive(i, j)
#                     if cnt < 2 or cnt > 3:
#                         board[i][j] = -1
#         for i in range(m):
#             for j in range(n):
#                 if board[i][j] == 2:
#                     board[i][j] = 1
#                 elif board[i][j] == -1:
#                     board[i][j] = 0
#
#         # 位运算
#         # 用当前位置数字的最后一位储存原来是否有细胞
#         # 前面几位组成的新数表示周围活细胞数目,即:
#         # 将周围细胞数目的值，左移一位
#         # 将这两个值做按位或运算
#         def count_cell(x, y):
#             points = [
#                 (x-1, y-1),
#                 (x-1, y),
#                 (x-1, y+1),
#                 (x, y-1),
#                 (x, y+1),
#                 (x+1, y-1),
#                 (x+1, y),
#                 (x+1, y+1),
#             ]
#             return sum((board[i][j] & 1) for i, j in points if 0 <= i < max_x and 0 <= j < max_y)
#
#         if not board:
#             return board
#
#         max_x, max_y = len(board), len(board[0])
#
#          # 计算周围细胞数目，并储存
#         for i in range(max_x):
#             for j in range(max_y):
#                 count = count_cell(i, j)
#                 count <<= 1
#                 board[i][j] |= count
#
#         for i in range(max_x):
#             for j in range(max_y):
#                 count = board[i][j] >> 1   # 右移一位，取出周围细胞数目
#                 board[i][j] &= 1   # 重新设置原先细胞状态
#                 if board[i][j] == 1:
#                     if count < 2 or count > 3:
#                         board[i][j] = 0
#                 else:
#                     if count == 3:
#                         board[i][j] = 1
#         return board
#
#
# solve = Solution()
# solve.gameOfLife(x)
# print(x)

# ############################### 290. 单词规律 #################################
# pattern = "abba"
# s = "dog cat cat dog"       # T
#
# # pattern = "abba"
# # s = "dog cat cat fish"      # false
#
# # pattern = "aaaa"
# # s = "dog cat cat dog"       # false
#
# # pattern = "abba"
# # s = "dog dog dog dog"     # false
#
#
# class Solution:
#     def wordPattern(self, pattern: str, str: str) -> bool:
#         str_list = str.split()
#         if len(pattern) != len(str_list):
#             return False
#         dic = {}
#         for i, ele in enumerate(pattern):
#             if ele not in dic:
#                 if str_list[i] not in dic.values():
#                     dic[ele] = str_list[i]
#                 else:
#                     return False
#             else:
#                 if dic[ele] != str_list[i]:
#                     return False
#         return True
#
#         # 利用map函数
#         # 巧用index函数返回找到第一个该元素的索引
#         res = str.split()
#         return list(map(pattern.index, pattern)) == list(map(res.index, res))
#
#
# solve = Solution()
# print(solve.wordPattern(pattern, s))
