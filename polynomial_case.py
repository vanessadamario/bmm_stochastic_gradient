import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures


def PolyFeatures_LinearParams(x_np_ply, y_np, k, bias=False):

    """
    Training of a linear model in the parameters, where the
    input is equivalent to the feature mapping (degree two)

                    Y = W Phi(X)

    Parameters:
        x_np_ply : feature map of input matrix, dimension n x d_ply, where n is
        the number of samples and d_ply is the number of feature (after mapping)
        y_np : output matrix, of dimension o x n
        k : number of points used in the learning phase
        bias : bool value, if True the constant term is added

    Return:
        curr_W : matrix of params, dimension o x d, using the linear NN, with
        polynomial mapping
        w.T : pseudo inverse solution

    We expect the two matrices to be the same in case of convergence of the NN
    """

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    n, d_ply = x_np_ply.shape
    o, _ = y_np.shape
    init_value = 1e-5
    grad_step = 5e-4
    n_iters = int(2e5)

    # initialization using one element of the data set
    # init = tf.constant(np.repeat(x_np_ply[0, :].reshape(1,d_ply), repeats=h, axis=0), dtype='float32')
    # variables relative to weights, small initialization numbers
    W = tf.Variable(init_value * tf.random_normal([o, d_ply]))

    linear_model = tf.matmul(W, tf.transpose(x))   # W_21 \dot XT

    if bias is True:                                # bias term added to the linear model
        b_np = np.random.randn(o, h)                # true value of b, components extracted from a normal distribution
        y_np += np.sum(b, axis=1)                   # y_:,i = (W_32 W_21 XT):,i + \sum_{i=1}^h B_:,i
        b = tf.Variable(init_value * tf.ones([o, d_ply]))
        linear_model += tf.reduce_sum(b, axis=1)

    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    optimizer = tf.train.GradientDescentOptimizer(grad_step)
    train = optimizer.minimize(loss)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(n_iters):
            sess.run(train, {x: x_np_ply[0:k, :], y: y_np[:, 0:k]})
            if (i % 50000 == 0):
                print("iteration", i)
                print("loss", sess.run(loss, {x:x_np_ply[0:k, :], y:y_np[:, 0:k]}))
        if bias is False:
            curr_W, curr_loss = sess.run([W, loss], {x: x_np_ply[0:k, :], y: y_np[:, 0:k]})
            print("loss function on the training set", curr_loss)
        else:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_np_ply[0:k, :], y: y_np[:, 0:k]})
        print("loss function of test set: ", sess.run(loss, {x:x_np_ply[k:, :], y:y_np[:, k:]}))

    # d_ply = x_np_ply.shape[1]                  # dimension of the feature vector

    alpha = np.dot(np.linalg.pinv(x_np_ply[0:k, :].dot(x_np_ply[0:k, :].T)), y_np[:, 0:k].T)
    w = np.dot(x_np_ply[0:k, :].T, alpha)
    print("distance between pseudoinverse and solution")
    print(np.linalg.norm(w.T - curr_W))

    return curr_W, w.T

################################################################################

def PolyFeatures_HiddenMapping(x_np, y_np, h, k, bias=False):

    """
    Training a neural network with one hidden layer, polynomial mapping
    performed in the hidden layer

                        Y = W2 ((W1 X) .* (W1 X))

    Initialization values for W1 correspond to row of the input matrix X
    Initialization values for W2 correspond to a random matrix (o x h)


    Parameters:
        x_np : matrix of input data, dimension n x d, with n number of samples
        y_np : matrix of output, dimension o x n
        h : int, dimension of hidden nodes
        bias : in True, the bias term is added
    Returns:
        curr_W1 : matrix from input to hidden layer
        curr_W2 : matrix from hidden to output layer
        rank_W1 : rank relative to the input layer transformation
        eigs_W1 : eigenvalues of the input layer transf
        rank_W1pow2 : rank of the product W1.T W1
        eigs_W1pow2 : eigenvalues of the product W1.T W1
    """

    grad_step = 2.5e-4    # learning rate
    init_val = 1e-3     # initialization value, used only in the bias term
    n_iters = int(4e5)  # number of iterations
    _, d = x_np.shape
    o, _ = y_np.shape

    init = init_val * tf.constant(np.repeat(x_np[0, :].reshape(1, d), repeats=h, axis=0), dtype='float32')
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    W1 = tf.Variable(init)
    W2 = tf.random_normal([o, h])
    linear_model = tf.matmul(W1, tf.transpose(x))
    linear_model = tf.matmul(W2, tf.multiply(linear_model, linear_model))

    if bias is True:                                # bias term added to the linear model
        b_np = np.random.randn(o, h)                # true value of b, components extracted from a normal distribution
        y_np += np.sum(b, axis=1)                   # y_:,i = (W_32 W_21 XT):,i + \sum_{i=1}^h B_:,i
        b = tf.Variable(init_val * tf.ones([o, h]))
        linear_model += tf.reduce_sum(b, axis=1)

    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    optimizer = tf.train.GradientDescentOptimizer(grad_step)
    train = optimizer.minimize(loss)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(n_iters):
            sess.run(train, {x: x_np[0:k, :], y: y_np[:, 0:k]})
            if (i%50000 == 0):
                print("iteration", i)
                print("loss", sess.run(loss, {x:x_np[0:k, :], y:y_np[:, 0:k]}))
        if bias is False:
            curr_W1, curr_W2, curr_loss = sess.run([W1, W2, loss], {x: x_np[0:k, :], y: y_np[:, 0:k]})
            print("loss function on the training set", curr_loss)

        else:
            curr_W1, curr_W2, curr_b, curr_loss = sess.run([W1, W2, b, loss], {x: x_np[0:k, :], y: y_np[:, 0:k]})
        print("loss function of test set: ", sess.run(loss, {x:x_np[k:, :], y:y_np[:, k:]}))

    rank_W1 =  np.linalg.matrix_rank(curr_W1)
    _, s, _ = np.linalg.svd(curr_W1)
    eigs_W1 = s

    rank_W1pow2 = np.linalg.matrix_rank(np.dot(curr_W1.T, curr_W1))
    _, s, _ = np.linalg.svd(np.dot(curr_W1.T, curr_W1))
    eigs_W1pow2 = s

    return curr_W1, curr_W2, rank_W1, eigs_W1, rank_W1pow2, eigs_W1pow2
################################################################################

def hidden_mapping(d_ply, w):

    """
    The function performs the Hadamard product between the elements of
    the inner matrix W, polynomial of degree 2
    Parameters:
        d_ply : dimension of the output matrix after the polynomial mapping
        w : matrix of dimension h x d
    Returns:
        square_matrix : matrix of dimension h x d_ply
        for each row w_h = w, which corresponds to a fixed h we get
        square_matrix[h, :] = w1**2, 2w1*w2, 2w1*w3, ..., w2**2, ..., wd**2
    """

    h, d = np.shape(w)
    square_matrix = np.zeros((h, d_ply))

    count = 0
    for i in range(d):
        for j in range(i, d):
            delta = (i == j)       # 0 if cross product, 1 otherwise
            square_matrix[:, count] = w[:, i] * w[:, j] * (2 - delta)
            count += 1

    return square_matrix


def main():

    n = 4                 # number of sample, for both training and test
    d = 7                 # input dimension (# of features)
    h = 5                 # number of hidden nodes
    o = 4                # dimension of the output, given a sample
    perc = 0.5            # percentage of data used in training
    bias = False
    degree = 2
    poly = PolynomialFeatures(degree, include_bias=True)
    k  = int(perc * n)    # number of training points

    ########################### DATA SET GENERATION ###########################
                            # Y = W2 ((W1 X).* (W1 X))
    x_np = np.random.randn(n, d)                      # input data
    w_1 = np.random.randn(h, d)
    w_2 = np.random.randn(o, h)                       # weights hidden to output
    y_np = w_2.dot((w_1.dot(x_np.T))**2)

    ############################# feature mapping #############################
    x_np_ply = poly.fit_transform(x_np)

    # in case where the bias is not included, we can get rid of all term of
    # degree less than 2 (bias, x1, ..., xd)
    x_np_ply = x_np_ply[:, d+1:]
    d_ply = x_np_ply.shape[1]                        # dim of the feature vector

    curr_W1, curr_W2, _, _, rank_W1pow2, eigs_W1pow2 = PolyFeatures_HiddenMapping(x_np, y_np, h, k)
    print("rank of inner matrix", rank_W1pow2)
    sol = curr_W2.dot(hidden_mapping(d_ply, curr_W1))
    print("rank of the solution", np.linalg.matrix_rank(sol))

    return


if __name__ == '__main__':
    main()
