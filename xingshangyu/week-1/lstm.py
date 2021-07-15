import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import sigmoid, tanh

def rn_init(shape):
	return tf.Variable(keras.initializers.random_normal()(shape=shape), trainable=True)

def repmat(x, l):
	return tf.repeat(tf.expand_dims(x, axis=0), [l], axis=0)


class LtsmUnit(keras.layers.Layer):
	def __init__(self, input_len, output_len, h_len, c_len, **kwargs):
		'''
		input_len: sample shape
		output_len: output shape
		h_len: length of hidden state vector h
		c_len: length of hidden state vector c
		'''
		super().__init__(**kwargs)
		i, o, h, c = input_len, output_len, h_len, c_len
		self.w = rn_init(shape=[c, i + h])
		self.wi = rn_init(shape=[c, i + h])
		self.wf = rn_init(shape=[c, i + h])
		self.wo = rn_init(shape=[c, i + h])
		self.w_out = rn_init(shape=[o, h])
		self.b = rn_init(shape=[c])
		self.bi = rn_init(shape=[c])
		self.bf = rn_init(shape=[c])
		self.bo = rn_init(shape=[c])
		self.b_out = rn_init(shape=[o])

	def calc(self, c_, h_, x):
		xh = tf.concat([x, h_], axis=-1)
		z = tanh(tf.matmul(xh, self.w, transpose_b=True) + self.b)
		zi = sigmoid(tf.matmul(xh, self.wi, transpose_b=True) + self.bi)
		zf = sigmoid(tf.matmul(xh, self.wf, transpose_b=True) + self.bf)
		zo = sigmoid(tf.matmul(xh, self.wo, transpose_b=True) + self.bo)
		c = zf * c_ + zi * z
		h = zo * tanh(c)
		y = sigmoid(tf.matmul(h, self.w_out, transpose_b=True) + self.b_out)
		return c, h, y


class LstmLayer(keras.layers.Layer):
	def __init__(self, layer_num, input_len, output_len, h_len, c_len, **kwargs):
		super().__init__(**kwargs)
		n, self.o, h, c = layer_num, output_len, h_len, c_len
		self.lays = [LtsmUnit(input_len, output_len, h_len, c_len, **kwargs) for i in range(n)]
		self.c0 = rn_init([c])
		self.h0 = rn_init([h])

	def call(self, x):
		n = len(self.lays)
		y = [tf.zeros(self.o) for i in range(n)]
		c, h = repmat(self.c0, len(x)), repmat(self.h0, len(x))
		for i in range(n):
			c, h, y[i] = self.lays[i].calc(c, h, x[:, i])
		return tf.concat(y, axis=-1)


def create_model(max_words, vocabulary_size, embed_mat) -> keras.Model:
	input = layers.Input((max_words, ))
	embed_size = tf.shape(embed_mat)[1]
	x = layers.Embedding(vocabulary_size, embed_size, embeddings_initializer=keras.initializers.Constant(embed_mat), trainable=False)(input)
	x = LstmLayer(layer_num=max_words, input_len=embed_size, output_len=64, h_len=64, c_len=64)(x)
	# x = layers.LSTM(128)(x)
	x = layers.Dense(1, activation='sigmoid')(x)
	model = keras.Model(inputs=input, outputs=x)
	model.compile(
		loss='binary_crossentropy',
		optimizer='adam',
		metrics=["accuracy"],
	)
	return model


if __name__ == '__main__':
	from data import load_data
	_, _, idx2vec = load_data(100, 5000)
	model = create_model(100, 5000, idx2vec)
	model.summary()