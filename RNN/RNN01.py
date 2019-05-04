from tensorflow_train import train02_RNN as tf
from tensorflow_train.train02_RNN import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
HIDDEN_LAYER = 128
BATCH_SIZE = 128
INPUTS_NUM = 28
STEPS_NUM = 28
CLASS_NUM = 10
learnRate = 0.001
traning_iter = 100000

x = tf.placeholder(tf.float32,shape=[None,INPUTS_NUM,STEPS_NUM])
y = tf.placeholder(tf.float32,shape=[None,CLASS_NUM])

weights = {
    'in':tf.Variable(tf.random_normal([INPUTS_NUM,HIDDEN_LAYER])),
    'out':tf.Variable(tf.random_normal(([HIDDEN_LAYER,CLASS_NUM])))
}
biases = {
    'in':tf.Variable(tf.random_normal([HIDDEN_LAYER,])),
    'out':tf.Variable(tf.random_normal(([CLASS_NUM,])))
}

#key of LSTM of RNN
def Rnn(X,weights,biases):
    X = tf.reshape(X,[BATCH_SIZE*STEPS_NUM,INPUTS_NUM])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in,[-1,STEPS_NUM,HIDDEN_LAYER])
    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_LAYER,forget_bias=1.0,state_is_tuple=True)
    _init_state = cell.zero_state(BATCH_SIZE,tf.float32)
    output,states = tf.nn.dynamic_rnn(cell,X_in,initial_state=_init_state,time_major=False)
    result = tf.matmul(states[1],weights['out'])+biases['out']
    return result

pred = Rnn(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels= y))
train_op = tf.train.AdamOptimizer(learnRate).minimize(cost)

correct = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*BATCH_SIZE<traning_iter:
          batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
          batch_xs = batch_xs.reshape([BATCH_SIZE,STEPS_NUM,INPUTS_NUM])
          sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
          if step % 20 == 0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
          step = step+1
