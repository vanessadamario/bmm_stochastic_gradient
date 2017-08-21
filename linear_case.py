import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


n = 20  # number of samples
d = 2  # input space dimension (variables)
train_dim = 0.5
k = int(n * train_dim)
x_np = np.random.randn(n, 2)

# w_12 = np.array([[2., 3., 2.], [0.3, 1., 0.]])
# matrix of weights (random numbers)
w_12 = np.array([[1., 0.], [5., 1.]])
w_32 = np.array([10., 1.])
b_1 = np.array([0., 0.])
b_2 = np.array([0., 0])

# linear case
y_np = w_32.dot(np.dot(w_12, x_np.T) + b_1.reshape(d, 1)*np.ones((d, n))) + b_2.dot(np.ones((d, n))) # output vector

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W1 = tf.Variable(1e-4*tf.ones([2, 2]))
W2 = tf.Variable(1e-4*tf.ones([1, 2]))
# b1 = tf.Variable(tf.ones([2, 1]))
# b2 = tf.Variable(tf.ones([1, 2]))


### fitting using the composition of the two linear transf

linear_model = tf.matmul(W1, tf.transpose(x)) # + b1*tf.ones([d, k])
linear_model = tf.matmul(W2, linear_model) # + tf.matmul(b2, tf.ones([d, k]))
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(100000):
        sess.run(train, {x: x_np[0:k, :], y: y_np[0:k]})

    # curr_W1, curr_W2, curr_b1, curr_b2, curr_loss = sess.run([W1, W2, b1, b2, loss], {x: x_np[0:k, :], y: y_np[0:k]})
    # print("W1: %s W2: %s b1: %s b2: %s loss: %s"%(curr_W1, curr_W2, curr_b1, curr_b2, curr_loss))

    curr_W1, curr_W2, curr_loss = sess.run([W1, W2, loss],
                                          {x: x_np[0:k, :], y: y_np[0:k]})
    print("W1: %s W2: %s loss: %s"%(curr_W1, curr_W2, curr_loss))
    print(sess.run(loss, {x:x_np[k:, :], y:y_np[k:]}))

print("real - predicted")
print(w_32.dot(w_12[: ,0]), curr_W2.dot(curr_W1[:, 0]))
print(w_32.dot(w_12[: ,1]), curr_W2.dot(curr_W1[:, 1]))
# print(w_32.dot(b_1), curr_W2.dot(curr_b1))

alpha = np.dot(np.linalg.pinv(x_np[0:k,:].dot(x_np[0:k,:].T)), y_np[0:k])
# alpha=(XXT)^{\pseudoInv}Y
w = np.dot(x_np[0:k,:].T, alpha)  # w = XT alpha
print("pseudo-inverse solution", w)


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
init_value = tf.constant(x_np[0, :].reshape((1, d)), dtype=tf.float32)
# init_value = tf.zeros([1, d])
W_single_layer = tf.Variable(init_value)  # vector of d dimensions #
# b_single_layer = tf.Variable(1e-4*tf.ones([1,1]))
single_layer_funct = tf.matmul(W_single_layer, tf.transpose(x)) # + b_single_layer

squared_deltas = tf.square(single_layer_funct - y)
loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(10000):
        sess.run(train, {x: x_np[0:k, :], y: y_np[0:k]})
        # print(sess.run(loss, {x:x_np[0:k, :], y: y_np[0:k]}))
    curr_W, curr_loss = sess.run([W_single_layer, loss],
                                {x: x_np[0:k, :], y: y_np[0:k]})
    print("W: %s loss: %s"%(curr_W, curr_loss))
    print(sess.run(loss, {x:x_np[k:, :], y:y_np[k:]}))

alpha = np.dot(np.linalg.pinv(x_np[0:k,:].dot(x_np[0:k,:].T)), y_np[0:k])
w = np.dot(x_np[0:k,:].T, alpha)  # w = XT alpha
print("pseudo-inverse solution", w)
