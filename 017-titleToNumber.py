# ############################### 174. 地下城游戏 ###############################
x = [[-2,-3, 3],
     [-5,-10,1],
     [10,30,-5]]        # 7

# x = [[2,-3, 3, 4],
#      [-5,-10,1,-3],
#      [10,30,-3,-5]]        # 7


class Solution:
    def calculateMinimumHP(self, dungeon):
        # # 这题不一样的地方在于必须要从终点往前推
        # dp = [[0] * len(dungeon[0]) for _ in range(len(dungeon))]
        # dp[-1][-1] = max(1, 1 - dungeon[-1][-1])
        # for j in range(2, len(dungeon[0]) + 1):
        #     dp[-1][-j] = max(1, dp[-1][-j + 1] - dungeon[-1][-j])
        #
        # for i in range(2, len(dungeon) + 1):
        #     dp[-i][-1] = max(1, dp[-i + 1][-1] - dungeon[-i][-1])
        #
        # for i in range(2, len(dungeon) + 1):
        #     for j in range(2, len(dungeon[0]) + 1):
        #         dp[-i][-j] = max(1, min(dp[-i + 1][-j], dp[-i][-j + 1]) - dungeon[-i][-j])
        # print(dp)
        # return dp[0][0]

        # 只用最后一行大小也能dp
        m, n = len(dungeon), len(dungeon[0])
        dp = [0] * n
        # 初始化最后一行
        dp[-1] = max(1, 1 - dungeon[-1][-1])
        for i in range(n - 2, -1, -1):
            dp[i] = max(1, dp[i + 1] - dungeon[-1][i])
        for j in range(m - 2, -1, -1):
            dp[-1] = max(1, dp[-1] - dungeon[j][-1])
            for k in range(n - 2, -1, -1):
                dp[k] = max(1, min(dp[k], dp[k + 1]) - dungeon[j][k])
        return dp[0]


solve = Solution()
print(solve.calculateMinimumHP(x))
