#encoding:utf-8
'''''
author:Wang junhe
time:2020.2.13 12:34
describe:Build a new neural_network
tips: self.num is a random list
      shape[1] is col,apply to col data, means how many col

'''''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import arange,random
#########random choose data.shape#########
class corresponding_choose(object):
    def __init__(self,data,n,m=1):
        self.num = arange(data.shape[m])
        random.shuffle(self.num)
        self.n=n
    def col(self,x_data):
        self.x = x_data[:,self.num[0:self.n]]
        return self.x
    def row(self,y_data):
        y = y_data[self.num[0:self.n],:]
        return y
    def col_2(self,x_data):
        self.x = x_data[::,self.num[0:self.n]]
        return self.x
    def row_2(self,y_data):
        y = y_data[self.num[0:self.n],::]
        return y