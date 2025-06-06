# -*- coding:utf-8 -*-
#!/usr/bin/python2
import networkx as nx


def diGraph_to_adj(diG, file):
    pathfile = file
    adj_dict = {}
    node_list = []
    adj_list = []
    adj = diG.adjacency()
    for x,y in adj:
        adj_dict[x] = y.keys()
    with open(pathfile, 'w') as f:
        for key in adj_dict:
            f.writelines(str(key) + ':' + str(adj_dict[key]))
            f.write('\n')


    return adj_dict

def resort_pathfile(pathfile):
    key_path_list = []
    with open(pathfile, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            # x = line.split(':')
            key_path_list.append(line)
    key_path_list.sort()
    with open(pathfile, 'w') as f2:
        for x in key_path_list:
            # x.strip('"')
            f2.writelines(str(x))
            f2.write('\n')

#SCS
def shortestCommonSupersequence(str1, str2):
    """
    :type str1: str
    :type str2: str
    :rtype: str
    """
    m = len(str1)
    n = len(str2)
    
    dp = [[''] * (n+1) for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + str1[i-1]
            else:
                if len(dp[i-1][j]) > len(dp[i][j-1]):
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i][j-1]
    
    i = 0
    j = 0
    lcs = dp[m][n]
    ans = ''
    for cur_char in lcs:
        while(i < m and str1[i] != cur_char):
            ans += str1[i]
            i += 1
        while(j < n and str2[j] != cur_char):
            ans += str2[j]
            j += 1
        
        ans += cur_char
        i += 1
        j += 1
    print(ans + str1[i:] + str2[j:])
    return ans + str1[i:] + str2[j:]