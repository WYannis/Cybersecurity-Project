# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:04:48 2019

@author: wyannis
"""

import pickle as pk
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc


data = pd.read_pickle('./alldata.txt')
data.head()

##Normalization
#We compress all the data in the range [0, 1]

X = data.drop(['Label'], axis=1)
X_data = (X - X.min())/(X.max() - X.min())
X_data.head()

X_data = X_data.drop(X_data.columns[X_data.isna().any().tolist()], axis=1)
# X_data.head()
X_data = np.float32(X_data)


##Auto-encoder
#We construct a 2-layer nn (encoder and decoder) to reconstruct the dataset. 
#Rationale: Anomalies will not be constructed as well as normal data

# Hyperparameters
learning_rate = 0.00001
num_steps = 30000
batch_size = 256
display_step = 1000
# examples_to_show = 10

#architecture
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 70

#Data matrix
# X_dataMatrix = X_data.as_matrix()
X_dataMatrix = np.float32(X_dataMatrix)
X_dataMatrix.shape

#Construct the graph 

#Define weight matrices
X = tf.placeholder(tf.float32, [None, num_input])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
} #initialize with normal variables

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

#Define encoding and decoding
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1 Wx+b
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#Define loss and optimizer

Y_pred = decoder_op
Y = X
diff = Y_pred - Y
loss = tf.reduce_mean(tf.square(diff))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#shuffle the data


#run the graph
sess = tf.Session()

sess.run(tf.global_variables_initializer())
#     sess.run(iterator.initializer)
cnt = 0
cntMax = X_dataMatrix.shape[0]/batch_size
for i in range(1, num_steps+1):
    cnt = cnt + 1
    if cnt > cntMax:
        np.random.shuffle(X_dataMatrix)
        cnt = 1
    _, l = sess.run([optimizer, loss],
                    feed_dict={X: X_dataMatrix[(cnt - 1) * batch_size : cnt * batch_size, :]})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
        
        
from sklearn.metrics import roc_curve, roc_auc_score, auc

#Difference between original data and reconstructed data
def rowError(X, X_pred):
    return np.square(X_pred - X).sum(axis=1)

#predict the new X
y = 1 - np.float32(data['Label'] == 'BENIGN')
X_pred = sess.run(decoder(encoder(X_data)))

#calculate the error
X_error = rowError(X_data, X_pred)

#Calculate ROC score
print('ROC score = ', roc_auc_score(y, X_error))

#Define ROC curve
fpr, tpr, thresholds = roc_curve(y, X_error)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


y_pred = np.float32(X_error > thresholds[np.argmax(tpr - fpr)])
y_pred


CM = confusion_matrix(y, y_pred)
#pas mal!


def accuracy(mat):
    return (mat[0][0] + mat[1][1])/mat.sum()
def precision(mat):
    return mat[0][0]/(mat[0][0] + mat[1][0])
def recall(mat):
    return mat[0][0]/(mat[0][0] + mat[0][1])

def show_result(mat):
    print('Accuracy = ' + str(accuracy(mat)) )
    print('Precision = ' + str(precision(mat)) )
    print('Recall = ' + str(recall(mat)) )
    
show_result(CM)