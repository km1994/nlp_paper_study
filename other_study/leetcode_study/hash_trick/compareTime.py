import time 
# 读取小文件
small_file = "small.txt"
small_list = []
small_set = set()
with open(small_file,"r",encoding="utf-8") as f:
    line = f.readline()
    while line:
        small_list.append(line)
        small_set.add(line)
        line = f.readline()

print("小样本情况下")
time_start=time.time()
print(f"200 in small_list：{200 in small_list}")
time_end=time.time()
print('程序运行时间:%s毫秒' % ((time_end - time_start)*1000))

time_start=time.time()
print(f"200 in small_set：{200 in small_set}")
time_end=time.time()
print('程序运行时间:%s毫秒' % ((time_end - time_start)*1000))


# 读取大文件
large_file = "large.txt"
large_list = []
large_set = set()
with open(large_file,"r",encoding="utf-8") as f:
    line = f.readline()
    while line:
        large_list.append(int(line))
        large_set.add(int(line))
        line = f.readline()

print("\n小样本情况下")
time_start=time.time()
print(f"10000 in large_list：{102146 in large_list}")
time_end=time.time()
print('程序运行时间:%s毫秒' % ((time_end - time_start)*1000))

time_start=time.time()
print(f"10000 in large_set：{102146 in large_set}")
time_end=time.time()
print('程序运行时间:%s毫秒' % ((time_end - time_start)*1000))