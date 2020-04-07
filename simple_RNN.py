'''''
tips:(1).Thinking about the shape of self.X, so self.x = reshape(self.X, [-1,chunk_size])
     (2).Beacuse the time we update parameter is when the  circul is end; Considering the originl neural_network,
         the 128 is the lenth of one hidden.
         the number of hidden is 28, this is also the number of time step, and the number of hidden layer's neural
     (3).Beacuse we need the sequence data, so we need to split the database which are Multiplicated by x and W as a sequence data.
     (4).如果 time_major == False (default), input的形状必须为 [batch_size, sequence_length, frame_size]
     !!!(5).ERROR:ValueError: sequence_length must be a vector of length batch_size, but saw shape: (2, 200, 128)
            SOLVE:initial_state=self.init_state
'''''
from tensorflow import placeholder,float32,transpose, nn, reduce_mean, multiply, log, reduce_sum, train
from add_layer import add_layer
from tensorflow.python.ops.rnn import dynamic_rnn
class simple_RNN(object):
    def __init__(self,chunk_size,n_chunk,hidden_chunk_size, batch_size,  n_class, learning_rate):
        self.X = placeholder(float32,[None,n_chunk,chunk_size])#200,28,28
        self.y = placeholder(float32,[None, n_class])

        self.LSTM_cell = nn.rnn_cell.BasicLSTMCell(hidden_chunk_size, forget_bias=1.0, state_is_tuple=True)#cotain state and hidden output
        self.init_state = self.LSTM_cell.zero_state(batch_size, float32)#shape is [200, 128] for h and c
        self.output, self.states = dynamic_rnn(self.LSTM_cell, self.X, initial_state=self.init_state, dtype=float32)#output contains all hidden_output list; states contains th finall time state and hidden_output

        self.result = add_layer(transpose(self.states[1]), hidden_chunk_size, n_class, activation_function=nn.softmax)

        self.loss = reduce_mean(-reduce_sum(multiply(transpose(self.y),log(self.result)),reduction_indices=[0]))
        self.train = train.AdamOptimizer(learning_rate).minimize(self.loss)











