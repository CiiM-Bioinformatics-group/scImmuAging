import os
import glob
import random
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import Counter


base_dir = "~/AgePrediction/data/processed/"
os.chdir(base_dir)

def parse_train(inputs, donor, age):
        train_cells = []
        train_labels = []
        class_map = {}
        for i in range(len(donor)):
                input_temp = inputs[inputs.donor_id == donor[i]]
                train_cells.append(np.array(input_temp.iloc[:, 2:]))
                train_labels.append(input_temp.iloc[0, 1])
                class_map[i] = age[i*100]
        
        return(
            np.array(train_cells),
            np.array(train_labels),
            class_map,
        )
class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg=0.00001):
                self.num_features = num_features
                self.l2reg = l2reg
                self.eye = tf.eye(num_features)
        def __call__(self, x):
                x = tf.reshape(x, (-1, self.num_features, self.num_features))
                xxt = tf.tensordot(x, x, axes=(2, 2))
                xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
                return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, activation='relu', padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.90)(x)
        return x


def dense_bn(x, filters):
        x = layers.Dense(filters, activation='relu')(x)#kernel_regularizer=regularizers.l2(0.01)
        x = layers.BatchNormalization(momentum=0.90)(x)
        return x

def residual_block(inputs, filters):
        x = conv_bn(inputs, filters)
        x = layers.Conv1D(filters, kernel_size = 1, padding="valid")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, inputs])
        x = layers.Activation('relu')(x)
        return x


def tnet(inputs, num_features):
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)
    
        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)    # 64
        #x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        #x = dense_bn(x, 256)
        x = dense_bn(x, 128)  ## 128
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

### to learn which channel/feature is better to the classification ###
class ChannelAttention(layers.Layer):
    def __init__(self):
        super(ChannelAttention, self).__init__()

    def build(self, input_shape):
        channels = input_shape[-1]

        #self.dense1 = layers.Conv1D(channels // 2, kernel_size=1, activation='relu', padding="valid")
        #self.dense2 = layers.Conv1D(channels, kernel_size=1, activation='sigmoid', padding="valid")
        
        self.dense1 = layers.Dense(channels // 2, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=[1])
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.expand_dims(x, axis=1)

        return inputs * x

initial_learning_rate = 0.001
decay_rate = 0.5

def lr_schedule(epoch):
    if epoch < 50:
        return initial_learning_rate
    elif epoch < 100:
        return initial_learning_rate * decay_rate
    else:
        return initial_learning_rate * decay_rate * decay_rate

learning_rate_fn = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


cell_type = "B"
NUM_CLASSES = 78
NUM_CELLS = 100
BATCH_SIZE = 32



trains_stable = pd.read_csv(base_dir + "/input_pn/" + cell_type + '_train_ordergene.txt', sep='\t')
trains = trains_stable.iloc[:, :]
tests_stable = pd.read_csv(base_dir + "/input_pn/" + cell_type + '_test_ordergene.txt', sep='\t')
tests = tests_stable.iloc[:, :]

train_data_mtx = trains.sort_values(by = ['age', 'donor_id']).reset_index(drop = True)     
test_data_mtx = tests.sort_values(by = ['age', 'donor_id']).reset_index(drop = True)   


train_input = train_data_mtx.iloc[:, 0:]
donor_train = np.array(list(set(train_input['donor_id'])))
age_train = train_input['age'].tolist()    


test_input = test_data_mtx.iloc[:, 0:]
donor_test = np.array(list(set(test_input['donor_id'])))
age_test = test_input['age'].tolist()    


train_cells, train_gs, CLASS_MAP1 = parse_train(
    train_input, donor_train, age_train
)


test_cells, test_gs, CLASS_MAP2 = parse_train(
    test_input, donor_test, age_test
)


train_dataset = tf.data.Dataset.from_tensor_slices((train_cells, train_gs))
test_dataset = tf.data.Dataset.from_tensor_slices((test_cells, test_gs))

train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
test_dataset = test_dataset.shuffle(buffer_size=len(test_dataset))

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

  
inputs = keras.Input(shape=(NUM_CELLS, train_cells.shape[2]))

x1 = conv_bn(inputs, 128)
x2 = conv_bn(x1, 64)
x3 = conv_bn(x2, 32)
x4 = conv_bn(x3, 32)
x4 = tnet(x4, 32)
x5 = conv_bn(x4, 32)
x5 = tf.concat([x5, x3], axis=-1)
x6 = conv_bn(x3, 64)
x6 = tf.concat([x6, x2], axis=-1)
x7 = conv_bn(x6, 128)    # 128
x7 = tf.concat([x7, x1], axis=-1)
x8 = conv_bn(x7, 256)
x9 = conv_bn(x8, 512)

x = layers.GlobalMaxPooling1D()(x9)
x = dense_bn(x, 256)  
x = layers.Dropout(0.5)(x)
x = dense_bn(x, 128)  
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="linear")(x)


model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate), metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.fit(train_dataset, epochs=200, validation_data=test_dataset, callbacks=[learning_rate_fn])


# for regression:
new_model = model                
preds = new_model.predict(test_cells)
preds[:, 0]
test_gs


np.sqrt(np.mean((test_gs-preds[:, 0])**2))
scipy.stats.pearsonr(test_gs, preds[:, 0])
     

evaluation = model.evaluate(test_cells, test_gs)
print(evaluation)

pre_df = {'age': test_gs, 'Prediction': preds[:,0]}
pre_df = pd.DataFrame(pre_df)
pre_df = pre_df.sort_values(by = ['age'])    
pre_df.to_csv("/home/wli/test.txt", index = False, sep = "\t")

