# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time


def test1():
  m1 = tf.constant([[3, 3]])
  m2 = tf.Variable([[2], [2]])

  prod = tf.matmul(m1, m2)

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    print(m2.eval())
    v = tf.assign(m2, tf.Variable([[3], [3]]))
    init = tf.global_variables_initializer()
    init.run()
    print(v.eval())
    # v = tf.assign(m2, tf.Variable([[4], [4]]))
    # print(v.eval())
    res = sess.run(prod)
    print(res)
    w = tf.constant(3)
    x = tf.placeholder(tf.float32, shape=[3, 1])
    w2 = tf.Variable(tf.random_uniform([4, 3], -1.0, 1.0))
    b2 = tf.Variable(tf.zeros([4, 1]))
    w3 = tf.Variable(tf.random_uniform([8, 4], -10.0, 80.0))
    b3 = tf.Variable(tf.zeros([8, 1]))
    w4 = tf.Variable([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    y = tf.matmul(w3, tf.sigmoid(tf.matmul(w2, x) + b2)) + b3
    y2 = tf.matmul(w2, x) + b2
    # init = tf.global_variables_initializer()
    # init.run()
    ym = tf.split(y, 8)
    yym = []
    for yi in ym:
      yym.append(tf.gradients(yi, w2))
    yy = tf.gradients(y, w2)
    init = tf.global_variables_initializer()
    init.run()
    # yy2 = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(y)
    # yy2.run()
    yy2 = tf.gradients(y2, x)

    res = sess.run(yy, feed_dict={x: [[1.34], [2.222], [3.67]]})
    res2 = sess.run(yy2, feed_dict={x: [[1.34], [2.222], [3.67]]})
    resm = []
    for yyi in yym:
      resm.append(sess.run(yyi, feed_dict={x: [[1.34], [2.222], [3.67]]})[0].tolist())
    # print(tf.transpose(res).eval())
    # print(resm)
    sumy = 0
    for i in range(8):
      print(resm[i][0][0])
      sumy += resm[i][0][0]
    print(res[0].tolist()[0])
    print(sumy)
    print(b3.eval())
    print(tf.scatter_nd_add(w4, np.array([[0]]), [[10, 20, 30]]).eval())
    print(res)
    # print(resm)
    # print(res * 0.02)
    # mm = tf.constant([[1,2],[3,4],[5,6]])
    # print(tf.reshape(mm,[3,2,1]).eval())
    # a = tf.constant([[1.2],[2.2]])
    # print(0.02*a)

    # print(w2.eval())
    # print(res2)
    embeddings = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0))
    init = tf.global_variables_initializer()
    init.run()
    # print(tf.nn.embedding_lookup(embeddings,[0,1,2]))
    # for i in tf.nn.embedding_lookup(embeddings,[0,1,2]):
    #  print(i)


def testGrad():
  x = tf.placeholder(tf.float32, shape=[None, 4, 1])
  a = tf.Variable(tf.eye(4, 4))
  b = tf.Variable(tf.eye(4, 1))
  y = tf.add(tf.matmul(a, x), b)
  with tf.Session() as sess:
    ys = tf.split(y, 4)
    tf.global_variables_initializer().run()
    m = tf.constant([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]], dtype=tf.float32)
    init_x = np.expand_dims(np.eye(2, 4), 2)
    sess.run(tf.gradients(tf.matmul(m, y), x), {x: init_x})
    sess.run(tf.gradients(ys[0], x), {x: init_x})
    sess.run(tf.gradients(ys[2], x), {x: init_x})


def mul_test():
  x = np.arange(1000 * 200 * 300).reshape([1000, 200, 300])
  y = np.arange(4000 * 200).reshape([4000, 200])
  tf_x = tf.constant(x)
  tf_y = tf.constant(y)
  # x = np.ndarray(shape=(10, 20, 30), dtype=float)
  # y = np.ndarray(shape=(30, 40), dtype=float)
  start = time.time()
  # res = tf.einsum('ijk,kl->ijl', tf_y, tf_x)
  # res = tf.einsum('ij,kjl->kil', tf_y, tf_x)
  for i in range(1000):
    # res = tf.matmul(tf_y,tf.reshape(tf_x,[200,300*1000]))
    res = tf.einsum('ij,kjl->kil', tf_y, tf_x)
  print(time.time() - start)
  start = time.time()
  # np.matmul(y,x)
  print(time.time() - start)
  # print(res.shape)


def broardcast_test():
  a = tf.constant(np.arange(10).reshape([5, 2, 1]))
  b = tf.constant(np.arange(10, 12).reshape([2, 1]))
  with tf.Session() as sess:
    print(sess.run(tf.add(a, b)))


def test_test():
  x = tf.placeholder(tf.float32, shape=[10, None])
  W = tf.Variable(tf.random_normal([20, 10]))
  b = tf.Variable(tf.zeros([20, 1]), dtype=tf.float32)
  y = tf.add(tf.matmul(W, x), b)
  v = tf.Variable(tf.zeros([20, 10]), dtype=tf.float32)
  v_holder = tf.placeholder(tf.float32, shape=[20, 10])
  op_v = v.assign_add(v_holder)
  x_val = np.arange(100).reshape([10, 10])

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    res = sess.run(y, feed_dict={x: x_val})
    sess.run(op_v, feed_dict={v_holder: res})
    # print(res.shape)
    # print(res[1,1])
    print(v[1, 1])
    print(type(res))


def test_sparse():
  xx = np.array([[[0, 0], [0, 1]], [[1, 0], [0, 0]], [[0, 1], [0, 0]]])
  yy = np.ones([3, 2, 1])
  print(np.matmul(xx, yy))
  x = tf.sparse_placeholder(tf.float32, shape=[None, 2, 2])
  y = tf.constant(np.ones([3, 2, 1]), dtype=tf.float32)
  # z = tf.matmul(x, y, a_is_sparse=True)
  # z = tf.sparse_tensor_dense_matmul(x,y)
  indices = [[1, 1, 1], [2, 0, 0], [3, 0, 1]]
  values = [1.0, 2.0, 3.0]
  dense_shape = [3, 2, 2]

  x_val = tf.SparseTensorValue(indices, values, dense_shape)
  zz = tf.matmul(x_val, y, a_is_sparse=True)
  with tf.Session() as sess:
    # sess.run(z, feed_dict={x: x_val})
    sess.run(zz)


def test_multiply():
  x = tf.constant(np.arange(10), dtype=tf.float32)
  y = tf.constant(np.arange(10, 20), dtype=tf.float32)
  indices = [3, 9, 1]
  values = [1.0, 3.0, 2.0]
  dense_shape = [10]

  x_val = tf.SparseTensorValue(indices, values, dense_shape)
  y_val = tf.sparse_to_dense(x_val.indices, x_val.dense_shape, x_val.values, validate_indices=False)
  with tf.Session() as sess:
    print(sess.run(tf.multiply(y_val, y)))


def test_tensorArray():
  x = tf.constant(np.ones([10, 4]), dtype=tf.float64)
  # arr = tf.TensorArray.split(x,[1]*10)
  ar = tf.TensorArray(dtype=tf.float64, size=10)

  with tf.Session() as sess:
    ar.write(0, x)
    print(ar.read(0).eval())


def test_split():
  x = tf.constant(np.arange(40).reshape([4, 10]), dtype=tf.float64)
  sp_list = tf.split(x, 10, 1)
  print(type(sp_list))
  with tf.Session() as sess:
    print(sp_list[0].eval())
    print(x[0].eval())


def test_gather():
  x = tf.constant(np.arange(10, 30).reshape(4, 5), dtype=tf.float64)
  y = tf.constant(np.arange(10,14).reshape(2,2),dtype=tf.int64)
  v = tf.Variable(np.ones([4,1]),dtype=tf.float64)
  v1 = tf.constant(np.ones([2,2]),dtype=tf.int64)
  v2 = tf.constant(np.ones([2,2]),dtype=tf.int64)
  i1 = tf.constant([3, 1], dtype=tf.int64)
  i2 = tf.constant([1, 3], dtype=tf.int64)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(tf.gather_nd(x, [[3, 1], [1, 3]]).eval())
    print(tf.gather(x, [i1, i2]).eval())
    print(tf.gather_nd(y,[[[0,1],[1,0]],[[1,1],[0,0]]]).eval())
    print(x[[1,2]].eval())
    print(tf.gather_nd(v,[[1],[0],[1]]).eval())
    print(tf.stack([v1,v2]).eval())

def test_sparse():
  a = tf.SparseTensor([[1,1]],[10.0],[3,3])
  b = tf.SparseTensor([[1, 1],[2,2]], [10,10], [3, 3])
  c = tf.ones([3,3])
  with tf.Session() as sess:
    print(tf.sparse_tensor_dense_matmul(a,c).eval())
    print((-1*c).eval())
    #print(tf.multiply(a,b))

def test_scatter_mul():
  a = tf.Variable([[1,2,3],[2,3,4],[3,4,5]],dtype=tf.int32)


  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    b = tf.scatter_mul(a,[0,1],[[-1,-1,-1],[-1,-1,-1]])
    c = tf.scatter_nd_add(a,[[0],[1]],[[-1,-1,-1],[-1,-1,-1]])
    d = tf.scalar_mul(0.1,a)
    print(d.eval())

def test_grad():
  var0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
  var1 = tf.Variable([3.0, 4.0], dtype=tf.float32)
  grads0 = tf.constant([0.1, 0.1], dtype=tf.float32)
  grads1 = tf.constant([0.01, 0.01], dtype=tf.float32)
  f = tf.add(var0,var1)
  opti = tf.train.GradientDescentOptimizer(3.0)
  sgd_op = opti.apply_gradients(zip([grads0, grads1], [var0, var1]))
  sgd_op2 = opti.apply_gradients(f)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sgd_op2.run()
    #print(list(zip([grads0, grads1], [var0, var1])))
    print(var0.eval())

def test_regularize():
  x = tf.Variable(np.ones([4,4]),dtype=tf.float32)
  y = tf.contrib.layers.l2_regularizer(0.1)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #print(list(zip([grads0, grads1], [var0, var1])))
    res = tf.contrib.layers.apply_regularization(y,[x])
    print(x.eval())

def test_lstm():
  lstm = tf.contrib.rnn.LSTMCell(100)
  random = tf.reshape(tf.tile(tf.random_uniform([ 200], -100, 100),[3]),[1,3,200])
  rand = tf.random_uniform([1,100],-50,50,dtype=tf.float32)
  rand = tf.zeros([1,100],dtype=tf.float32)
  x= tf.Variable(random,dtype=tf.float32)
  #c_state = tf.Variable(tf.zeros([1,100]),dtype=tf.float32)
  #m_state = tf.Variable(tf.zeros([1, 100]), dtype=tf.float32)
  #print(lstm.state_size)
  #state = tf.ones([1,100])
  with tf.Session() as sess:
    #val, state = tf.nn.dynamic_rnn(lstm, x)
    #cell,_ = tf.contrib.rnn.MultiRNNCell([lstm] * 1)
    tf.global_variables_initializer().run()
    #print(lstm.state_size)
    output, out_state = tf.nn.dynamic_rnn(lstm, x,initial_state=tf.contrib.rnn.LSTMStateTuple(rand,rand))
    #output, out_state = tf.contrib.rnn.static_rnn(lstm, [x],dtype=tf.float32)
    #output,state = lstm.(x,state)
    #res = sess.run(output)
    #res2 = sess.run(out_state)
    #output, out_state = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32,initial_state=res2)
    tf.global_variables_initializer().run()
    print(sess.run(output))
    #print(sess.run(out_state))
    #print(res2)
    #print(output)

if __name__ == '__main__':
  # mul_test()
  # broardcast_test()
  # test_test()
  # print(np.__version__)
  # test_sparse()
  # test_multiply()
  # test_tensorArray()
  # test_split()
  #test_gather()
  # test_test()
  # test_sparse()
  # test_scatter_mul()
  #test_grad()
  #test_regularize()
  test_lstm()


