# -*- coding: utf-8 -*-
import tensorflow as tf

m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[2]])

prod = tf.matmul(m1,m2)

sess = tf.Session()

res = sess.run(prod)
print(res)

sess.close()


