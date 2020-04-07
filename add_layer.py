#encoding:utf-8
'''''
author:Wang junhe
time:2020,2,12 18:45
describe:add_layer
study_by:莫烦Python
tips:  1. sess.run(initialize_all_variables())
       2. eights = Variable(random_normal([out_size,in_size]))
       3. def add_layer(input,in_size,out_size,activation_function=None):
       4. a=add_layer(data,3,2,sigmod)
'''''
from tensorflow import random_normal,Variable,zeros,matmul,transpose,nn
def add_layer(input,in_size,out_size,activation_function=None):
        Weights = Variable(random_normal([out_size,in_size]))
        Biases = Variable(zeros([out_size,1])+0.1)
        Wx_plus_b = matmul(Weights,input)+Biases
        if activation_function==None:
            output = Wx_plus_b
        else:
            output = activation_function(transpose(Wx_plus_b))
            output = transpose(output)
        return output


def add_layer_dropput(input, in_size, out_size,keep_prob = None,activation_function=None):
    Weights = Variable(random_normal([out_size, in_size]))
    Biases = Variable(zeros([out_size, 1]) + 0.1)
    Wx_plus_b = matmul(Weights, input) + Biases
    Wx_plus_b = nn.dropout(Wx_plus_b,keep_prob= keep_prob)
    if activation_function == None:
        output = Wx_plus_b
    else:
        output = activation_function(transpose(Wx_plus_b))
        output = transpose(output)
    return output




