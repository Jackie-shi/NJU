import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(max_words, vocabulary_size, embed_mat) -> keras.Model:
	input = layers.Input((max_words, ))
	embed_size = tf.shape(embed_mat)[1]
	x = layers.Embedding(vocabulary_size, embed_size, embeddings_initializer=keras.initializers.Constant(embed_mat), trainable=False)(input)
	x = layers.LSTM(128)(x)
	x = layers.Dense(128, activation='relu')(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(1, activation='sigmoid')(x)
	model = keras.Model(inputs=input, outputs=x)
	model.compile(
		loss='binary_crossentropy',
		optimizer='adam',
		metrics=["accuracy"],
	)
	return model