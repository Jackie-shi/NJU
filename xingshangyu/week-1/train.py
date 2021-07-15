# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras
from data import load_data
# from cnn import create_model
from lstm import create_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train_eval(model: keras.Model, x_train, y_train, x_test, y_test):
	model.fit(x_train, y_train, 64, 4, validation_split=0.1)
	scores = model.evaluate(x_test, y_test, verbose=0)
	print('Test accuracy:', scores[1])


max_word_length = 64
vocabulary_size = 50000
def main():
	(x_train, y_train), (x_test, y_test), embed_mat = load_data(max_word_length, vocabulary_size)
	model = create_model(max_word_length, vocabulary_size, embed_mat)
	model.summary()
	train_eval(model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
	main()