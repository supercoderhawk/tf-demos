#-*- coding: UTF-8 -*-
from HiddenMarkovModel import HiddenMarkovModel
import numpy as np

def test():
  i = np.array([[0.6], [0.4], [0.1], [0.2]])
  t = np.array(np.arange(16), dtype=np.float64).reshape([4, 4])
  o = np.array(np.arange(40000), dtype=np.float64).reshape([4, 10000]).T
  ob = np.array(np.arange(10000), dtype=np.int32)
  model = HiddenMarkovModel(t, o, i)
  states_seq, state_prob = model.run_viterbi(ob)

if __name__ == '__main__':
  test()