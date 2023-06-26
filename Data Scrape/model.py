#!/usr/bin/env python

import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf, keras
from base64 import b64encode
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import colors
from keras.layers.normalization import BatchNormalization

X = np.load("scaled_data.npy")
y = np.load("binary_labels.npy")

## Use for multi-class classification
# y = tf.keras.utils.to_categorical(
#     y, num_classes=5, dtype='float32'
# )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1) 





def LSTM(n_classes, sequence_size, n_features):
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(28, input_shape=(sequence_size, n_features), recurrent_dropout=0.1))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    if n_classes == 1:
        model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
    else:
        model.add(keras.layers.Dense(n_classes, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


sequence_size = 28
n_features = 2
n_classes = 1

model = LSTM(n_classes, sequence_size, n_features)
model.summary()

checkpoint_path = "{dir}/2_class.ckpt".format(dir='./')

latest = tf.train.latest_checkpoint('./')

if latest:
    model.load_weights(latest)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                             save_weight_only=True,
                                             save_best_only=True)



callbacks = [cp_callback]


epochs = 50
batch_size = 128

history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                   epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                   verbose=1)

model.save("2_class")
model.save("2_class.h5")

# It can be used to reconstruct the model identically.
# my_model = keras.models.load_model("my_model")

ypred=model.predict(X_test)
print(classification_report(y_test, (ypred>0.5)))
print(confusion_matrix(y_test, (ypred>0.5)))
print(f'Test Accuracy -> {accuracy_score(y_test, (ypred>0.5))}')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('2_class_accuracy.png')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('2_class_loss.png')


