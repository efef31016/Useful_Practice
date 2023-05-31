
def power_set(s):
    n = len(s)
    power_set_size = 2**n
    result = []
    
    # 迭代 0 到 2^n
    for i in range(power_set_size):
        subset = []
        for j in range(n):
            #  i 的二進位表示中的第 j 位是否為 1
            if (i & (1 << j)) > 0:
                print(i, j)
                subset.append(s[j])
        
        print(subset)
        result.append(subset)
    
    return result


def generate_binary_lists(n):
    if n == 0:
        return [[]]
    
    prev_lists = generate_binary_lists(n - 1)
    result = []
    
    for lst in prev_lists:
        result.append(lst + [0])
        result.append(lst + [1])
    
    return result
