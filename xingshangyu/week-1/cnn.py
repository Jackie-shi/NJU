# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


filter_num = 256
min_filter = 3
max_filter = 6
def create_model(max_words, vocabulary_size, embed_mat) -> keras.Model:
	embed_size = tf.shape(embed_mat)[1]
	input = layers.Input((max_words, ))
	x = layers.Embedding(vocabulary_size, embed_size, embeddings_initializer=keras.initializers.Constant(embed_mat), trainable=False)(input)
	tmp = [x for i in range(max_filter - min_filter)]
	for i in range(min_filter, max_filter):
		tmp[i - min_filter] = layers.Conv1D(filter_num, i, activation='relu')(x)
		tmp[i - min_filter] = layers.GlobalMaxPooling1D()(tmp[i - min_filter])
	x = tf.concat(tmp, axis=1)
	x = layers.Dense(256, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
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
