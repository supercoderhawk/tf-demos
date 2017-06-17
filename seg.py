# -*- coding: UTF-8 -*-
import tensorflow as tf
import math
import pandas as pd
import numpy as np
from prepare_data import PrepareData
from config import CorpusType

zy = {'be': 0.5,
      'bm': 0.5,
      'eb': 0.5,
      'es': 0.5,
      'me': 0.5,
      'mm': 0.5,
      'sb': 0.5,
      'ss': 0.5
      }
zy = {i: np.log(zy[i]) for i in zy.keys()}


def viterbi(nodes):
  """
  维特比译码：除了第一层以外，每一层有4个节点。
  计算当前层（第一层不需要计算）四个节点的最短路径：
     对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
     上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
     paths 采用字典的形式保存（路径：路径长度）。
     一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
  """
  paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
  for layer in range(1, len(nodes)):  # 后面的每一层
    paths_ = paths.copy()  # 先保存上一层的路径
    # node_now 为本层节点， node_last 为上层节点
    paths = {}  # 清空 path
    for node_now in nodes[layer].keys():
      # 对于本层的每个节点，找出最短路径
      sub_paths = {}
      # 上一层的每个节点到本层节点的连接
      for path_last in paths_.keys():
        if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
          sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[path_last[-1] + node_now]
      # 最短路径,即概率最大的那个
      sr_subpaths = pd.Series(sub_paths)
      sr_subpaths = sr_subpaths.sort_values()  # 升序排序
      node_subpath = sr_subpaths.index[-1]  # 最短路径
      node_value = sr_subpaths[-1]  # 最短路径对应的值
      # 把 node_now 的最短路径添加到 paths 中
      paths[node_subpath] = node_value
  # 所有层求完后，找出最后一层中各个节点的路径最短的路径
  sr_paths = pd.Series(paths)
  sr_paths = sr_paths.sort_values()  # 按照升序排序
  return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def viterbi_new(emission, A, init_A, return_score=False):
  """
  维特比算法的实现，所有输入和返回参数均为numpy数组对象
  :param emission: 发射概率矩阵，对应于本模型中的分数矩阵，4*length
  :param A: 转移概率矩阵，4*4
  :param init_A: 初始转移概率矩阵，4
  :param return_score: 是否返回最优路径的分值，默认为False
  :return: 最优路径，若return_score为True，返回最优路径及其对应分值
  """
  tags_count = 4
  length = emission.shape[1]
  path = np.ones([tags_count, length], dtype=np.int32) * -1
  corr_path = np.zeros([length], dtype=np.int32)
  path_score = np.ones([tags_count, length], dtype=np.float64) * (np.finfo('f').min / 2)
  path_score[:, 0] = init_A + emission[:, 0]

  for pos in range(1, length):
    for t in range(tags_count):
      for prev in range(tags_count):
        temp = path_score[prev][pos - 1] + A[prev][t] + emission[t][pos]
        if temp >= path_score[t][pos]:
          path[t][pos] = prev
          path_score[t][pos] = temp

  max_index = np.argmax(path_score[:, -1])
  corr_path[length - 1] = max_index
  for i in range(length - 1, 0, -1):
    max_index = path[max_index][i]
    corr_path[i - 1] = max_index
  if return_score:
    return corr_path, path_score[max_index, -1]
  else:
    return corr_path


def read_dictionary(dict_path):
  dict_file = open(dict_path, 'r', encoding='utf-8')
  dict_content = dict_file.read().splitlines()
  dictionary = {}
  dict_arr = map(lambda item: item.split(' '), dict_content)
  for _, dict_item in enumerate(dict_arr):
    dictionary[dict_item[0]] = int(dict_item[1])
  dict_file.close()
  if len(dictionary) < vocab_size:
    return None
  else:
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for i in range(vocab_size, len(dictionary)):
      dictionary.pop(reverse_dictionary[i])
  return dictionary


vocab_size = 4000  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embed_size = 100  # 字向量长度
hidden_units = 150  # 隐含层节点数
# layer_num = 2        # bi-lstm 层数
# concat_embed_size = window
alpha = 0.1
tags_count = 5
# lr = tf.placeholder(tf.float32)
batch_length = 40
batch_size = 20  # tf.placeholder(tf.int32)  # 注意类型必须为 tf.int32
dtype = tf.float32

embeddings = tf.Variable(
  tf.truncated_normal([vocab_size, embed_size], stddev=-1.0 / math.sqrt(embed_size), dtype=dtype),
  name='embeddings')
w = tf.Variable(tf.truncated_normal([hidden_units, tags_count], stddev=1.0 / math.sqrt(embed_size), dtype=dtype),
                name='w')
b = tf.Variable(tf.zeros([tags_count], dtype=dtype), name='b')
input = tf.placeholder(tf.int32, shape=[None])
input_embeds = tf.expand_dims(tf.nn.embedding_lookup(embeddings, input),0)
lstm = tf.contrib.rnn.LSTMCell(hidden_units)
lstm_output, lstm_out_state = tf.nn.dynamic_rnn(lstm, input_embeds, dtype=dtype)
lstm_output = tf.reshape(lstm_output, [-1, hidden_units])
word_score = tf.matmul(lstm_output, w) + b
dictionary = read_dictionary('corpus/pku_dict.utf8')


def sentence2index(sentence):
  index = []
  for word in sentence:
    if word not in dictionary:
      index.append(0)
    else:
      index.append(dictionary[word])

  return index


def tags2words(sentence, tags):
  words = []
  word = ''
  for tag_index, tag in enumerate(tags):
    if tag == 0:
      words.append(sentence[tag_index])
    elif tag == 1:
      word = sentence[tag_index]
    elif tag == 2:
      word += sentence[tag_index]
    else:
      words.append(word + sentence[tag_index])
      word = ''
  # 处理最后一个标记为I的情况
  if word != '':
    words.append(word)

  return words


saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, 'tmp/lstm-cross-entropy3.ckpt')
  # sentence = '我爱北京天安门'
  # sentence = '小明是南京师范大学的学生'
  sentence = '迈向充满希望的新世纪'
  seq = sentence2index(sentence)
  scores = sess.run(word_score, feed_dict={input: seq}).T[:4,:]
  transition = np.array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 0, 0]])
  transition_init = np.array([0.5, 0.5, 1, 1])
  path = viterbi_new(scores, transition, transition_init)
  print(path)
  print(tags2words(sentence, path))


def seg(sentence):
  with tf.Session() as sess:
    saver.restore(sess, 'tmp/lstm-cross-entropy9.ckpt')
    # sentence = '我爱北京天安门'
    # sentence = '小明是南京师范大学的学生'
    # sentence = '迈向充满希望的新世纪'
    seq = sentence2index(sentence)
    scores = sess.run(word_score, feed_dict={input: seq}).T[:4, :]
    transition = np.array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 0, 0]])
    transition_init = np.array([0.5, 0.5, 1, 1])
    path = viterbi_new(scores, transition, transition_init)
    # print(path)
    # print(tags2words(sentence, path))
    return path

def estimate_cws(current_labels, correct_labels):
  cor_dict = {}
  curt_dict = {}
  curt_start = 0
  cor_start = 0
  for label_index, (curt_label, cor_label) in enumerate(zip(current_labels, correct_labels)):
    if cor_label == 0:
      cor_dict[label_index] = label_index + 1
    elif cor_label == 1:
      cor_start = label_index
    elif cor_label == 3:
      cor_dict[cor_start] = label_index + 1

    if curt_label == 0:
      curt_dict[label_index] = label_index + 1
    elif curt_label == 1:
      curt_start = label_index
    elif curt_label == 3:
      curt_dict[curt_start] = label_index + 1

  cor_count = 0
  recall_length = len(curt_dict)
  prec_length = len(cor_dict)
  for curt_start in curt_dict.keys():
    if curt_start in cor_dict and curt_dict[curt_start] == cor_dict[curt_start]:
      cor_count += 1

  return cor_count, prec_length, recall_length

def evaluate_model():
  pre = PrepareData(4000, 'pku', dict_path='corpus/pku_dict.utf8', type=CorpusType.Test)
  sentences = pre.raw_sentences
  labels = pre.labels_index
  corr_count = 0
  re_count = 0
  total_count = 0

  for _, (sentence, label) in enumerate(zip(sentences, labels)):
    tag = seg(sentence)
    cor_count, prec_count, recall_count = estimate_cws(tag, np.array(label))
    corr_count += cor_count
    re_count += recall_count
    total_count += prec_count
  prec = corr_count / total_count
  recall = corr_count / re_count

  print(prec)
  print(recall)
  print(2 * prec * recall / (prec + recall))

evaluate_model()