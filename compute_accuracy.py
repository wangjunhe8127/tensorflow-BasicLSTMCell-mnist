#encoding:utf-8
'''''
author:Wang junhe 
time:4:13 pm Saturday, feb. 15th, 2020
describe:compute_accuracy
tips: in this ,we can use reduce_mean
'''''
from tensorflow import equal,argmax,float32,cast,reduce_mean
def compute_accuracy(prediction,v_ys):
    correct_prediction = equal(argmax(prediction,0),argmax(v_ys,0))
    accuracy = reduce_mean(cast(correct_prediction,float32))
    return accuracy