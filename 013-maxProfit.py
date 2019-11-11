# ############################### 121. 买卖股票的最佳时机 ###############################
# x = [7,1,5,3,6,4]       # 5
# x = []       # 5
# # x = [7,6,4,3,1]       # 0
#
#
# class Solution(object):
#     def maxProfit(self, prices):
#         """
#         :type prices: List[int]
#         :rtype: int
#         """
#         # 用dp的思想,每一步找到最大
#         min_p, max_p = float('inf'), 0
#         for i in range(len(prices)):
#             min_p = min(min_p, prices[i])
#             max_p = max(max_p, prices[i] - min_p)
#         return max_p
#
#
#         # # 超时
#         # profit = 0
#         # for i_in, i_price in enumerate(prices[:-1]):
#         #     o_price = max(prices[i_in+1:])
#         #     profit = max(profit, (o_price - i_price))
#         # return profit
#
#
# solve = Solution()
# print(solve.maxProfit(x))

# ############################### 122. 买卖股票的最佳时机 II ###############################
# x = [7,1,5,3,6,4]       # 7
# x = [1,2,3,4,5]         # 4
# x = [7,1,2,1,7]         # 7
# # x = [7,6,4,3,1]       # 0
#
#
# class Solution(object):
#     def maxProfit(self, prices):
#         """
#         :type prices: List[int]
#         :rtype: int
#         """
#         profit = 0
#         for i in range(1, len(prices)):
#             tmp = prices[i] - prices[i - 1]
#             if tmp > 0:
#                 profit += tmp
#         return profit
#
#
# solve = Solution()
# print(solve.maxProfit(x))

# ############################### 123. 买卖股票的最佳时机 III ###############################
x = [3,3,5,0,0,3,1,4]       # 6
x = [1,2,3,4,5]         # 4
x = [7,6,4,3,1]       # 0
x = [1,2,4,2,5,7,2,4,9,0]   # 13


class Solution(object):
    # def insert_profit(self, profit, tmp_profit):
    #     if not profit:
    #         profit.append(tmp_profit)
    #     elif len(profit) == 1:
    #         if tmp_profit > profit[0]:
    #             profit.append(tmp_profit)
    #         else:
    #             profit.insert(0, tmp_profit)
    #     else:
    #         if tmp_profit > profit[0]:
    #             profit.pop(0)
    #             if tmp_profit > profit[0]:
    #                 profit.append(tmp_profit)
    #             else:
    #                 profit.insert(0, tmp_profit)
    #
    # def maxProfit(self, prices):
    #     """
    #     :type prices: List[int]
    #     :rtype: int
    #     """
    #     profit = []
    #     tmp_profit = 0
    #     for i in range(1, len(prices)):
    #         tmp = prices[i] - prices[i - 1]
    #         if tmp > 0:
    #             tmp_profit += tmp
    #         else:
    #             self.insert_profit(profit, tmp_profit)
    #             tmp_profit = 0
    #
    #
    #             # if not profit:
    #             #     profit.append(tmp_profit)
    #             # elif len(profit) == 1:
    #             #     if tmp_profit > profit[0]:
    #             #         profit.append(tmp_profit)
    #             #     else:
    #             #         profit.insert(0, tmp_profit)
    #             # else:
    #             #     if tmp_profit > profit[0]:
    #             #         profit.pop(0)
    #             #         if tmp_profit > profit[1]:
    #             #             profit.append(tmp_profit)
    #             #         else:
    #             #             profit.insert(0, tmp_profit)
    #             # tmp_profit = 0
    #
    #     if tmp_profit > 0:
    #         self.insert_profit(profit, tmp_profit)
    #
    #     print(profit)
    #     return sum(profit)

    def maxProfit(self, prices):
        if not prices: return 0
        n = len(prices)
        dp = [[0] * n for _ in range(3)]
        for k in range(1, 3):
            pre_max = -prices[0]
            for i in range(1, n):
                pre_max = max(pre_max, dp[k - 1][i - 1] - prices[i])
                dp[k][i] = max(dp[k][i - 1], prices[i] + pre_max)
        return dp[-1][-1]



solve = Solution()
print(solve.maxProfit(x))
