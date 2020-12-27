# 读取大文件
large_file = "large.txt"
large_list = []
large_dict = dict()
with open(large_file,"r",encoding="utf-8") as f:
    line = f.readline()
    while line:
        large_list.append(int(line))
        large_dict[int(line)] = len(large_list) 
        line = f.readline()

key = 3000
# 利用 列表 查询 3000 的 索引
for i in range(len(large_list)):
    if key == large_list[i]:
        print(f"Use list select {key} key is {i+1}")
        break

# 利用 哈希表 查询 3000 的 索引
print(f"Use Dict select {key} key is {large_dict[key]}")


# leetCode 205. 同构字符串
print("leetCode 205. 同构字符串")

class Solution1():
    def isIsomorphic(self,s,t):
        s_len = len(s)
        t_len = len(t)
        if s_len!=t_len:
            return False
        dic = {}
        for i in range(s_len):
            if s[i] in dic and dic[s[i]]!=t[i]:
                return False
            else:
                dic[s[i]]=t[i]
        return True 

solution = Solution1()
print(f"egg and add is Isomorphic:{solution.isIsomorphic('egg','add')}")   
print(f"foo and bar is Isomorphic:{solution.isIsomorphic('foo','bar')}")   
print(f"paper and title is Isomorphic:{solution.isIsomorphic('paper','title')}")   

# leetCode 387. 字符串中的第一个唯一字符
print("leetCode 387. 字符串中的第一个唯一字符")

class Solution2():
    def firstUniqChar(self,s):
        dic = {}
        for c in s:
            if c not in dic:
                dic[c]=0
            dic[c] +=1
        for i, c in enumerate(s):
            if dic[c] == 1:
                return i
        return -1

solution = Solution2()
print(f"leetcode the firstUniqChar:{solution.firstUniqChar('leetcode')}")   
print(f"loveleetcode the firstUniqChar:{solution.firstUniqChar('loveleetcode')}")  


        


