from keras import models
from keras import layers
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

# custom R2-score metrics for keras backend
#https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )




train_data_dir = '/shared/btc-transactions/train/train'
#validation_data_dir = '/shared/btc-transactions/validation/validation'

train_df = pd.read_csv('/home/hallgato-horvathkristof/Transaction_graphs/Inceptionv3/Volatile/df_vol_train.csv')
valid_df = pd.read_csv('/home/hallgato-horvathkristof/Transaction_graphs/Inceptionv3/Volatile/df_vol_valid.csv')


datagen=ImageDataGenerator(rescale=1./255.,
                           rotation_range=40,
                           zoom_range=0.1,
			                )   #validation split!

#datagen=ImageDataGenerator(rescale=1./255.,)


img_width, img_height = 380, 380


train_generator=datagen.flow_from_dataframe(
dataframe=train_df,
directory=train_data_dir,
x_col="block_heights",
y_col="volatilites",
has_ext=False,      #x_col column doesnt has the file extensions
#subset="training",     if validation split is set in ImageDataGenerator
batch_size=16,
seed=42,
shuffle=True,
class_mode="other",  #for regression other should be used
target_size=(img_width, img_height))



valid_generator=datagen.flow_from_dataframe(
dataframe=valid_df,
directory=train_data_dir,
x_col="block_heights",
y_col="volatilites",
has_ext=False,              #x_col column doesnt has the file extensions
#subset="validation",      if validation split is set in ImageDataGenerator
batch_size=16,
seed=42,
shuffle=True,
class_mode="other",
target_size=(img_width, img_height))




conv_base = InceptionV3(weights=None, include_top=False, input_shape=(img_width,img_height,3))

model = models.Sequential()
model.add(conv_base)
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(Dense(1))
model.add(Dense(1, activation='linear'))

conv_base.trainable = True

#error at callbacks if the learning rate is explicitly set somewhere
#sgd = SGD(lr=0.01, decay=1e-7, momentum=.9)
rms = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss='mse', optimizer=rms, metrics=['mae', r2_keras])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


filepath_loss = '/home/hallgato-horvathkristof/Transaction_graphs/Inceptionv3/Volatile/Inceptionv3_smaller_Volatile_DataAug_best_loss.hdf5'
filepath_csv = '/home/hallgato-horvathkristof/Transaction_graphs/Inceptionv3/Volatile/Inceptionv3_smaller_Volatile_DataAug_logs.csv'

callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=3, min_lr=0.001),
            ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
            CSVLogger(filepath_csv, separator = ",", append = False)]

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50,
                    callbacks=callbacks,
                   )
