import torch
import numpy as np
import regex as re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import random

def split(kg):
    # train/test and train/validation splits
    
    train = []
    test = []
    valid = []
    
    for target_cluster in kg[1]:
        if len(target_cluster) > 0:
            (random.shuffle(target_cluster))

            train_split, test_split = np.split(target_cluster,
                                        [int(0.8*len(target_cluster))])
            train_split, valid_split = np.split(train_split,
                                          [int(0.8*len(train_split))])
            train.append(train_split)
            test.append(test_split)
            valid.append(valid_split)
        
    return train,test,valid

def node_ind(train,test,valid):
    # extract node index based on subset
    
    train_nodes = []
    test_nodes = []
    val_nodes = []
    for i in train:
        if len(i)>0:
            for j in i:
                j = str(j)
                # print(j)
                ss = re.findall('<(.*?)>', j)
                ssh = str(ss[0])
                target = re.findall('\d+',ssh)
                target = int(target[0])
                train_nodes.append(target)
    for i in test:
        if len(i)>0:
            for j in i:
                j = str(j)
                ss = re.findall('<(.*?)>', j)
                ssh = str(ss[0])
                target = re.findall('\d+',ssh)
                target = int(target[0])
                test_nodes.append(target)
    for i in valid:
        if len(i)>0:
            for j in i:
                j = str(j)
                ss = re.findall('<(.*?)>', j)
                ssh = str(ss[0])
                target = re.findall('\d+',ssh)
                target = int(target[0])
                val_nodes.append(target)
    return train_nodes, test_nodes, val_nodes


def labels(kg):
    # extract node labels
    
    labels = []
    
    for i in kg[1]:
        for j in i:
            j = str(j)
            ss = re.findall('<(.*?)>', j)
            ssh = str(ss[0])
            sst = str(ss[-1])
            target = re.findall('\d+',ssh)

            target = int(target[0])
            label = re.findall('\d+',sst)
            label = int(label[0])
            labels.append((target, label))

    l = sorted(labels, key=lambda x: x[0])
    nodes = [x[0] for x in l]
    labs = [x[1] for x in l]
    labs = torch.tensor(labs, dtype=torch.int64)

    return labs, nodes


def masks(nodes,train,test,valid):
    # generate subset masks for training/evaluation
    
    tr = []
    tes = []
    val = []
    for i in train:
        tr.append((int(i),True))
        tes.append((int(i),False))
        val.append((int(i),False))
    
    for i in test:
        tr.append((int(i),False))
        tes.append((int(i),True))
        val.append((int(i),False))
    
    for i in valid:
        tr.append((int(i),False))
        tes.append((int(i),False))
        val.append((int(i),True))
    
    tr_sorted = sorted(tr, key=lambda x: x[0])
    tr_mask = [n[1] for n in tr_sorted]
    
    tes_sorted = sorted(tes, key=lambda x: x[0])
    tes_mask = [n[1] for n in tes_sorted]
        
    val_sorted = sorted(val, key=lambda x: x[0])
    val_mask = [n[1] for n in val_sorted]

    tr_mask = torch.tensor(tr_mask, dtype=torch.bool)
    tes_mask = torch.tensor(tes_mask,dtype=torch.bool)
    val_mask = torch.tensor(val_mask,dtype=torch.bool)
    return tr_mask, tes_mask, val_mask

def class_dist(labels,tr_mask,tes_mask,val_mask):
    # plot class distributions of the data
    
    train_labs = []
    for i in range(len(tr_mask)):
        if tr_mask[i]==True:
            train_labs.append(int(labels[i]))

    test_labs = []
    for i in range(len(tes_mask)):
        if tes_mask[i]==True:
            test_labs.append(int(labels[i]))
    
    val_labs = []
    for i in range(len(val_mask)):
        if val_mask[i]==True:
            val_labs.append(int(labels[i]))
    
    df_train = pd.DataFrame(train_labs, columns=['class'])
    df_tes = pd.DataFrame(test_labs, columns=['class'])
    df_val = pd.DataFrame(val_labs, columns=['class'])
    tr = (df_train['class'].sort_values().value_counts(sort=False))
    test = (df_tes['class'].sort_values().value_counts(sort=False))
    val = (df_val['class'].sort_values().value_counts(sort=False))
    
    plotdata = pd.DataFrame({
    "train": tr.values,
    "test":test.values,
    "valid":val.values
    }, 
    index=['0','1','2','3','4','5','6'])
    plotdata.plot(kind="bar")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.title("Class distribution of node labels")
    plt.xlabel("Class")
    plt.ylabel("Total nodes")
    
    # plt.bar(train_labs)
    plt.show()

def edges_info(kg,nodes):
    # extract edge info for training (not currently used)
    
    heads=[]
    tails=[]
    edges = []
    edge_feat = []

    for i in kg[0]:
        i = str(i)
        ss = re.findall('<(.*?)>', i)
        s = re.findall('"(.*?)"', i)
        ss2 = str(ss)

        x = re.findall('edge_\d+',ss2)
        if len(x)>0:
            h = str(ss[0])
            ed = str(ss[1])
            t = str(ss[2])
            head = re.findall('\d+',h)
            tail = re.findall('\d+',t)
            edg = re.findall('\d+',ed)
            heads.append(int(head[0]))
            tails.append(int(tail[0]))
            edge_feat.append(int(edg[0]))
            
    edge_feats = torch.tensor(edge_feat, dtype=torch.int64)
    heads = np.array(heads)
    tails = np.array(tails)
    ind = np.stack((heads,tails))
    edge_index = torch.tensor(ind, dtype=torch.int64)

    return edge_index, edge_feats    