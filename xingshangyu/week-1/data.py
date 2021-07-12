# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np


def shuffle2(x, y):
	p = np.random.permutation(x.shape[0])
	return x[p], y[p]

longest = 0
embed_size = 50
def load_data(max_words, vocabulary_size, test_split = 0.1):
	global longest
	word2idx = {}
	idx2vec = np.zeros((vocabulary_size, embed_size))
	with open('dataset/word2vec/glove.6B.%dd.txt' % embed_size) as w2v_file:
		for i in range(vocabulary_size - 1):
			line = w2v_file.readline()
			idx = line.find(' ')
			word = line[:idx]
			word2idx[word] = i + 1
			line = line[idx + 1:]
			if line[-1] == '\n':
				line = line[:-1]
			line = '[' + line + ']'
			idx2vec[i + 1] = eval(line.replace(' ', ','))
	def get_word_idx(word):
		try:
			ret = word2idx[word]
		except KeyError:
			ret = 0
		return ret

	with open('dataset/rt-polaritydata/rt-polarity.pos', encoding='utf-8', mode='r') as pos_file:
		pos = pos_file.readlines()
	with open('dataset/rt-polaritydata/rt-polarity.neg', encoding='utf-8', mode='r') as neg_file:
		neg = neg_file.readlines()
	pl = len(pos)
	nl = len(neg)
	data_size = pl + nl
	x = np.zeros((data_size, max_words), dtype='uint32')
	pos += neg
	for i in range(data_size):
		words = pos[i].split(' ')
		longest = max(longest, len(words))
		for j in range(min(len(words), max_words)):
			word = words[j]
			if word:
				x[i][j] = get_word_idx(word)
	
	test_size = int(data_size * test_split) // 2
	y_test = np.append(np.ones(test_size, dtype='uint8'), np.zeros(test_size, dtype='uint8'))
	y_train = np.append(np.ones(pl - test_size, dtype='uint8'), np.zeros(len(neg) - test_size, dtype='uint8'))
	x_test = np.append(x[:test_size], x[pl: pl + test_size], axis=0)
	x_train = np.append(x[test_size: pl], x[pl + test_size:], axis=0)
	return shuffle2(x_train, y_train), shuffle2(x_test, y_test), idx2vec


if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test), idx2vec = load_data(100, 5000)
	print(longest)
