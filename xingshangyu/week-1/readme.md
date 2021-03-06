# 第一周

使用cnn和lstm进行文本情感二分类

## 目录结构

- dataset/
  - rt-polaritydata/
    - rt-polarity.neg
    - rt-polarity.pos
  - word2vec/
    - glove.6B.50d.txt
    - glove.6B.100d.txt
    - glove.6B.200d.txt
    - glove.6B.300d.txt
- cnn.py
- lstm.py
- data.py
- train.py
- log.txt

## 训练及评估

训练模型并评估结果(环境为tf2)：

`python3 train.py > log.txt`

## 运行结果

`cat log.txt`:

### cnn

```
~/Workspace/research/nlp-summer-camp/1/cnn » cat log.txt                                                                       xsy@ASUS-VivoBook
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to   
==================================================================================================
input_1 (InputLayer)            [(None, 64)]         0              
__________________________________________________________________________________________________
embedding (Embedding)           (None, 64, 50)       2500000     input_1[0][0]  
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 62, 256)      38656       embedding[0][0]  
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 61, 256)      51456       embedding[0][0]  
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 60, 256)      64256       embedding[0][0]  
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 256)          0           conv1d[0][0]   
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 256)          0           conv1d_1[0][0]   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 256)          0           conv1d_2[0][0]   
__________________________________________________________________________________________________
tf.concat (TFOpLambda)          (None, 768)          0           global_max_pooling1d[0][0]   
                                                                 global_max_pooling1d_1[0][0]   
                                                                 global_max_pooling1d_2[0][0]   
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          196864      tf.concat[0][0]  
__________________________________________________________________________________________________
dropout (Dropout)               (None, 256)          0           dense[0][0]  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            257         dropout[0][0]  
==================================================================================================
Total params: 2,851,489
Trainable params: 351,489
Non-trainable params: 2,500,000
__________________________________________________________________________________________________
Epoch 1/4
135/135 [==============================] - 4s 20ms/step - loss: 0.7160 - accuracy: 0.5601 - val_loss: 0.5658 - val_accuracy: 0.7146
Epoch 2/4
135/135 [==============================] - 2s 14ms/step - loss: 0.5528 - accuracy: 0.7219 - val_loss: 0.5358 - val_accuracy: 0.7198
Epoch 3/4
135/135 [==============================] - 2s 14ms/step - loss: 0.4805 - accuracy: 0.7716 - val_loss: 0.5134 - val_accuracy: 0.7406
Epoch 4/4
135/135 [==============================] - 2s 15ms/step - loss: 0.4055 - accuracy: 0.8226 - val_loss: 0.5199 - val_accuracy: 0.7365
Test accuracy: 0.7420262694358826
```

### lstm

```
~/Workspace/research/nlp-summer-camp/multimodel-sentiment-analysis/xingshangyu/week-1(main*) » cat log.txt                                    xsy@ASUS-VivoBook
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 64)]              0   
_________________________________________________________________
embedding (Embedding)        (None, 64, 50)            2500000   
_________________________________________________________________
lstm_layer (LstmLayer)       (None, 4096)              2150528   
_________________________________________________________________
dense (Dense)                (None, 1)                 4097  
=================================================================
Total params: 4,654,625
Trainable params: 2,154,625
Non-trainable params: 2,500,000
_________________________________________________________________
Epoch 1/4
135/135 [==============================] - 21s 41ms/step - loss: 0.7275 - accuracy: 0.5050 - val_loss: 0.6362 - val_accuracy: 0.6240
Epoch 2/4
135/135 [==============================] - 4s 28ms/step - loss: 0.6170 - accuracy: 0.6663 - val_loss: 0.6332 - val_accuracy: 0.6635
Epoch 3/4
135/135 [==============================] - 4s 28ms/step - loss: 0.5379 - accuracy: 0.7302 - val_loss: 0.6069 - val_accuracy: 0.6667
Epoch 4/4
135/135 [==============================] - 4s 28ms/step - loss: 0.4886 - accuracy: 0.7629 - val_loss: 0.6477 - val_accuracy: 0.6604
Test accuracy: 0.6669793725013733
```
