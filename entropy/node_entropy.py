#encoding: utf-8
import copy
import csv

import numpy as np

# number of motif

Nm = 8


def countMotifs(A,nodN):
    node_motifs_list=['' for i in range(nodN)]
    Nm_1=count_chain(A,nodN,2,motif=1,node_motifs_list=node_motifs_list)
    Nm_2=count_chain(A,nodN,3,motif=2,node_motifs_list=node_motifs_list)
    Nm_3 = count_triangle(A, nodN,motif=3,node_motifs_list=node_motifs_list)
    Nm_4=count_chain(A,nodN,4,motif=4,node_motifs_list=node_motifs_list)
    Nm_5=count_star(A,nodN,3,motif=5,node_motifs_list=node_motifs_list)
    Nm_6=count_qua(A,4,motif=6,node_motifs_list=node_motifs_list)
    Nm_7=count_chain(A,nodN,5,motif=7,node_motifs_list=node_motifs_list)
    Nm_8=count_star(A,nodN,4,motif=8,node_motifs_list=node_motifs_list)
    num=[Nm_1,Nm_2,Nm_3,Nm_4,Nm_5,Nm_6,Nm_7,Nm_8]
   # print ('count_motifs: '+str(num))
    return num,node_motifs_list

def count_star(A,N,neiN,motif,node_motifs_list):
    print('开始计算motif：', motif)
    n = 0
    a = copy.copy(A)
    for i in range(N):
        print('搜索star中心：',i)
        if (np.sum(a[i]) >= neiN ):
            node_motifs_list[i] += str(motif)
            print('该节点可以作为中心：',i)
            n += 1
            edge_num = neiN
            while edge_num > 0:
                for k in range(len(a[i])):
                    if a[i][k] > 0:
                        node_motifs_list[k] += str(motif)
                        print('star角为：',k)
                        edge_num -= 1
            for j in range(i):
                a[N - j - 1][i] = 0
            x = np.nonzero(a[i])
            nei_Index = x[0][:neiN]
            a[i].fill(0)
            for j in nei_Index:
                a[j].fill(0)
                for k in range(N):
                    a[k][j] = 0
    return n


def find_next(a,N,i,rest,stack):
    if rest==0:
        a[i].fill(0)
        for j in range(N):
            a[j][i] = 0
        print('当前搜索完成！')
        return i
    else:
        if np.sum(a[i])>0:
            for j in range(N):
                a[j][i]=0
            x = np.nonzero(a[i])
            a[i].fill(0)
            next_Index=x[0][0]
            stack.append(next_Index)
            print('继续搜索节点：', next_Index)
            return find_next(a,N,next_Index,rest-1,stack)
        else:
            p=stack.pop()
            print('弹出节点：',p)
            return -1


def count_chain(A,N,len,motif,node_motifs_list):
    print('开始计算motif：',motif)
    n=0
    a = copy.copy(A)
    stack=[]
    for i in range(N):
        #当前搜索的起点是i，入栈
        print('当前搜索起点：',i)
        stack.append(i)
        if find_next(a,N,i,len-1,stack)>=0:
            n+=1
            for j in stack:
                node_motifs_list[j]+=str(motif)
        p=stack.pop()
        print('弹出节点：', p)
    return n


def count_triangle(A,N,motif,node_motifs_list):
    print('开始搜索motif：',motif)
    a = copy.copy(A)
    for i in range(N):
        print('搜索起点为：',i)
        for j in range(i, N):
            if a[i][j] > 0:
                for k in range(j, N):
                    if a[j][k] > 0 and a[k][i] > 0:
                        print('形成三角形：{} -> {} -> {}'.format(i,j,k))
                        node_motifs_list[i]+=str(motif)
                        node_motifs_list[j] += str(motif)
                        node_motifs_list[k] += str(motif)
    return 0


def count_qua(A, N,motif,node_motifs_list):
    print('开始搜索motif：', motif)
    a = copy.copy(A)
    for i in range(N):
        print('搜索起点为：', i)
        for j in range(i,N):
            if a[i][j]>0:
                for k in range(j,N):
                    if a[j][k]>0:
                        for l in range(k,N):
                            if a[k][l]>0 and a[l][i]>0:
                                print('形成四边形：{} -> {} -> {} -> {}'.format(i, j, k,l))
                                node_motifs_list[i] += str(motif)
                                node_motifs_list[j] += str(motif)
                                node_motifs_list[k] += str(motif)
                                node_motifs_list[l] += str(motif)
    return 0


A=[]
adjfile= open('../data/cora/adj.csv', "r")
lines = adjfile.readlines()
for line in lines:
    A_line=[]
    for item in line:
        A_line.append(int(item))
    A.append(A_line)

num,node_motifs=countMotifs(A,len(A))
print('num:',num)
print('node_motifs:',node_motifs)


