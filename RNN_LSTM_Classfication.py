#encoding:utf-8
'''''
author:Wang Junhe
time:4:28 pm Sunday, March 29th, 2020
describe:RNN_LSTM_biseLSTM
         The data is mnist, the order is row pixel
         Beacuse the task is classfication,so all the sequence can be useful-->zi = 1,zf = 1
             Beacuse we only need the last output, so wo can set zo = 1, and use the last on
        sess.run(global_variables_initializer()) is not     sess.run(tables_initializer())
             
trips:
'''''
from tensorflow import keras, Session, transpose, global_variables_initializer
from modular.simple_RNN import simple_RNN
from modular.compute_accuracy import compute_accuracy
from modular.random_choose import corresponding_choose
(train_x_image, train_y), (test_x_image, test_y) = keras.datasets.mnist.load_data(path='/home/xiaoshumiao/.keras/datasets/mnist.npz')
train_y = keras.utils.to_categorical(train_y)
test_y=keras.utils.to_categorical(test_y)

epochs = 1000
n_classes = 10
batch_size = 200#number
chunk_size = 28
n_chunk = 28
rnn_size = 128#the lenth of a hidden_neural_layer or the number of hidden_neural
learning_rate = 0.001
rnn = simple_RNN(chunk_size, n_chunk, rnn_size, batch_size, n_classes, learning_rate)

with Session() as sess:
    sess.run(global_variables_initializer())
    for i in range(epochs):
        train_data = corresponding_choose(train_x_image, batch_size, m=0)
        train_x_betch = train_data.row_2(train_x_image) / 255.
        train_y_betch = train_data.row(train_y)
        sess.run(rnn.train,feed_dict={rnn.X:train_x_betch,rnn.y:train_y_betch})
        if i % 20 ==0:
            test_data = corresponding_choose(test_x_image, 200, m=0)
            test_x_betch = test_data.row_2(test_x_image) / 255.
            test_y_betch = test_data.row(test_y)
            b = sess.run(rnn.result, feed_dict={rnn.X: test_x_betch})
            c = sess.run(compute_accuracy(b, transpose(test_y_betch)))
            print(c)