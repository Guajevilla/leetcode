# ############################### 331. 验证二叉树的前序序列化 ################################
x = "9,3,4,#,#,1,#,#,2,#,6,#,#"         # T
x = "1,#"                               # F
x = "9,#,#,1"                           # F


class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        lis = preorder.split(',')
        return lis


solve = Solution()
print(solve.isValidSerialization(x))
