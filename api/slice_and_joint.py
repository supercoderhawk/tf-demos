import tensorflow as tf

class SliceAndJoint(object):
  def __init__(self):
    self.sess = tf.Session()



sess = tf.Session()
params = tf.constant([6, 3, 4, 1, 5, 9, 10])
params2 = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
indices = tf.constant([[1,1]])
#s = tf.Tensor(dtype=tf.int32, value_index=[1])
output = tf.gather(params2, 1)
ss = sess.run(output)
print(sess.run(tf.add(params2[2], tf.gather(params2, 2))))
sess.close()

