#encoding: utf-8
import copy
import csv
from entropy.Entropy import graphEntropy
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
    #print('开始计算motif：', motif)
    n = 0
    a = copy.copy(A)
    for i in range(N):
        #print('搜索star中心：',i)
        if (np.sum(a[i]) >= neiN ):
            node_motifs_list[i] += str(motif)
           # print('该节点可以作为中心：',i)
            n += 1
            edge_num = neiN
            while edge_num > 0:
                for k in range(len(a[i])):
                    if a[i][k] > 0:
                        node_motifs_list[k] += str(motif)
                        #print('star角为：',k)
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
        #print('当前搜索完成！')
        p = stack.pop()
        #print('弹出末尾节点：', p)
        return True
    else:
        if np.sum(a[i])>0:
            x = np.nonzero(a[i])
            for next_Index in x[0]:
                if  next_Index not in stack:
                    stack.append(next_Index)
                    #print('搜索到了：', next_Index)
                    return find_next(a,N,next_Index,rest-1,stack)

        else:
            p=stack.pop()
            #print('弹出穷途节点：',p)
            return False


def count_chain(A,N,len,motif,node_motifs_list):
    #print('开始计算motif：',motif)
    n=0
    a = copy.copy(A)
    for i in range(N):
        #print('当前搜索起点：',i)
        stack = []
        stack.append(i)
        if find_next(a,N,i,len-1,stack):
            n+=1
            for j in stack:
                node_motifs_list[j]+=str(motif)
    #print('chain: {}  has {}'.format(motif,n))
    #chain都被计算了两次，头尾各一次
    return n


def count_triangle(A,N,motif,node_motifs_list):
    n=0
    #print('开始搜索motif：',motif)
    a = copy.copy(A)
    for i in range(N):
        #print('搜索起点为：',i)
        for j in range(i, N):
            if a[i][j] > 0:
                for k in range(j, N):
                    if a[j][k] > 0 and a[k][i] > 0:
                        #print('形成三角形：{} -> {} -> {}'.format(i,j,k))
                        node_motifs_list[i]+=str(motif)
                        node_motifs_list[j] += str(motif)
                        node_motifs_list[k] += str(motif)
                        n+=1
    return n


def count_qua(A, N,motif,node_motifs_list):
    n=0
    #print('开始搜索motif：', motif)
    a = copy.copy(A)
    for i in range(N):
        #print('搜索起点为：', i)
        for j in range(i,N):
            if a[i][j]>0:
                for k in range(j,N):
                    if a[j][k]>0:
                        for l in range(k,N):
                            if a[k][l]>0 and a[l][i]>0:
                                #print('形成四边形：{} -> {} -> {} -> {}'.format(i, j, k,l))
                                node_motifs_list[i] += str(motif)
                                node_motifs_list[j] += str(motif)
                                node_motifs_list[k] += str(motif)
                                node_motifs_list[l] += str(motif)
                                n+=1
    return n


def save_list(filename,l):
    with open(filename, 'w') as file_object:
        for item in l:
            file_object.write(str(item)+'\n')
    return file_object.close()



def nodeEntropy(graphEntropy,node_motifs_list,count_motifs):
    motif_node_number = [2,3,3,4,4,4,5,5]
    node_entropys=[0 for i in range(len(node_motifs_list))]
    node_index = 0
    for node_numbers in node_motifs_list:
        if node_numbers=='':
            continue
        node_sum=0
        for number in node_numbers:
            motif_num = int(number)
            index = motif_num - 1
            #某种类型的entropy，分配到所有motif上，再分配到点上
            node_sum += graphEntropy[index] / (count_motifs[index]*motif_node_number[index])
        node_entropys[node_index]=node_sum
        node_index += 1
    return node_entropys


A=[]
adjfile= open('../data/cora/adj.csv', "r")
lines = adjfile.readlines()
for line in lines:
    vector = [float(x) for x in line.split(",")]
    A_line=[]
    for item in vector:
        A_line.append(round(item))
    A.append(A_line)
A=np.array(A)
nodN=len(A)
count_motifs,node_motifs=countMotifs(A,nodN)
save_list('./count_motifs',count_motifs)
save_list('./node_motifs',node_motifs)
graph_entropy=graphEntropy(count_motifs,nodN)
save_list('./graph_entropy',graph_entropy)
node_entropy=nodeEntropy(graph_entropy,node_motifs,count_motifs)
save_list('node_entropy',node_entropy)


