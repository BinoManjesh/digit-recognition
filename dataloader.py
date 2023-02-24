import pandas as pd
import numpy as np

def store(w1,b1):
    w=w1.reshape(1,w1.size)
    b=b1.reshape(1,b1.size)
    df = pd.DataFrame(w)
    df2 = pd.DataFrame(b)
    df.to_csv('weights.csv',mode='w',header=False,index=False)
    df2.to_csv('bias.csv',mode='w',header=False,index=False)

def getwmean():
    df = pd.read_csv('weights.csv',header=None)
    wmean = df.to_numpy()
    return wmean

def getbmean():
    df = pd.read_csv('bias.csv',header=None)
    bmean = df.to_numpy()
    return bmean

def loadtrain():
    df = pd.read_csv('train.csv')
    label1 = df.filter(['label'])
    df.drop('label', axis=1, inplace=True)
    label = label1.to_numpy()
    train = df.to_numpy()
    train=train.astype(np.float)
    arr = np.zeros((len(df),10))
    for i in range(len(df)):
        arr[i][label[i]]=1
    train/=255.0
    return train, arr

def main():
    print(loadtrain())