# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import math
import time

decay = 0.85
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len = 32  # 句子长度
vocab_size = 4000  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embed_size = 100  # 字向量长度
class_num = 5
hidden_units = 150  # 隐含层节点数
# layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）
# concat_embed_size = window
alpha = 0.1
tags_count = 5
# lr = tf.placeholder(tf.float32)
batch_length = 40
keep_prob = tf.placeholder(tf.float32)
batch_size = 200  # tf.placeholder(tf.int32)  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置
dtype = tf.float32

# X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
# y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')
embeddings = tf.Variable(
  tf.truncated_normal([vocab_size, embed_size], stddev=-1.0 / math.sqrt(embed_size), dtype=dtype),
  name='embeddings')
w = tf.Variable(tf.truncated_normal([hidden_units, tags_count], stddev=1.0 / math.sqrt(embed_size), dtype=dtype),
                name='w')
b = tf.Variable(tf.zeros([tags_count], dtype=dtype), name='b')
input = tf.placeholder(tf.int32, shape=[batch_size, batch_length])
label = tf.placeholder(tf.int32, [batch_size, batch_length])
input_embeds = tf.reshape(tf.nn.embedding_lookup(embeddings, input), [batch_size, batch_length, embed_size])
lstm = tf.contrib.rnn.LSTMCell(hidden_units)
lstm_output, lstm_out_state = tf.nn.dynamic_rnn(lstm, input_embeds, dtype=dtype)
lstm_output = tf.reshape(lstm_output, [-1, hidden_units])
word_score = tf.matmul(lstm_output, w) + b
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(label,[-1]), logits=word_score))

tvars = tf.trainable_variables()
# 获取损失函数对于每个参数的梯度
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
# 梯度下降计算
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
character_batches = np.load('corpus/pku_training_character_batches.npy')
label_batches = np.load('corpus/pku_training_label_batches.npy')

saver = tf.train.Saver(max_to_keep=100)

epoches = 10
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  last_time = time.time()
  for i in range(epoches):
    for index, (character_batch, label_batch) in enumerate(zip(character_batches, label_batches)):
      sess.run(train_op, feed_dict={input: character_batch, label: label_batch})
    print(time.time()-last_time)
    last_time  = time.time()
    saver.save(sess,'tmp/lstm-cross-entropy%d.ckpt'%i)
