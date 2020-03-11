'''
@Project:DeepLearningAction

@Author:lincoln

@File:dnn

@Time:2020-03-11 09:29:28

@Description:使用tensorflow低级api训练DNN手写数字识别
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#定义隐藏层
def neuron_layer(layer_name, input_data, neuron_num, activation=None):
    with tf.name_scope(layer_name):
        input_dim = int(input_data.get_shape()[1])

        stddev = 2/np.sqrt(input_dim)
        init = tf.truncated_normal((input_dim, neuron_num),stddev=stddev)

        w = tf.Variable(init,name="weights")
        b = tf.Variable(tf.zeros([neuron_num]), name="bias")
        z = tf.matmul(input_data, w) +b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

#定义节点数量参数
n_input = 28*28
n_hidden1= 300
n_hidden2 = 100
n_output = 10

#定义网络结构-->定义损失函数-->定义优化方法-->执行

#网络输入
x = tf.placeholder(tf.float32, shape=(None, n_input), name="x")
y = tf.placeholder(tf.int64, shape=(None), name="y")

#网络结构
with tf.name_scope("dnn"):
    hidden1 = neuron_layer("hidden1", x, n_hidden1,activation="relu")
    hidden2 = neuron_layer("hidden2", hidden1,n_hidden2, activation="relu")
    logits = neuron_layer("logits", hidden2, n_output)

#损失函数
with tf.name_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

#网络优化
lr = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train = optimizer.minimize(loss)

#定义评估方法
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#执行
n_epochs = 400
batch_size = 50
minist = input_data.read_data_sets("../datasets/MINIST")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(minist.train.num_examples//batch_size):
            x_batch, y_batch = minist.train.next_batch(batch_size)
            y_batch_one_hot = np.eye(10)[y_batch]
            sess.run(train,feed_dict={x:x_batch, y:y_batch_one_hot})
        acc_train = accuracy.eval(feed_dict={x:x_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={x:minist.test.images,
                                            y:minist.test.labels})
        print(epoch, "acc_train:", acc_train, "acc_test:", acc_test)
    save_path = saver.save(sess,"./my_model_final.ckpl")

