# ############################### 165. 比较版本号 ###############################
version1 = "0.1"
version2 = "1.1"        # -1

# version1 = "1.0.1"
# version2 = "1"          # 1

# version1 = "7.5.2.4"
# version2 = "7.5.3"      # -1

# version1 = "1.01"
# version2 = "1.001"      # 0

# version1 = "1.0"
# version2 = "1.0.0"      # 0

# version1 = "1.0"
# version2 = "1.0"        # 0

# version1 = "1"
# version2 = "1.1"        # -1


class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        v1 = version1.split('.')
        v2 = version2.split('.')
        i = 0
        while i < len(v1) and i < len(v2):
            if int(v1[i]) > int(v2[i]):
                return 1
            elif int(v1[i]) < int(v2[i]):
                return -1
            i += 1

        while i < len(v1):
            if int(v1[i]) != 0:
                return 1
            i += 1
        while i < len(v2):
            if int(v2[i]) != 0:
                return -1
            i += 1

        return 0

        # import itertools
        # for x, y in itertools.zip_longest(version1.split("."), version2.split("."), fillvalue=0):
        #     if int(x) != int(y): return 1 if int(x) > int(y) else -1
        # return 0


solve = Solution()
print(solve.compareVersion(version1, version2))
