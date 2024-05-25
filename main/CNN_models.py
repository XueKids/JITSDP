from keras import regularizers
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential


def SimpleNet():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(14, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.03)))
    model.add(Dense(1, activation='sigmoid'))

    return model

def DropNet():
    # create model
    model = Sequential()
    model.add(Conv1D(32, 3, padding="same", input_shape=(14, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.2))  #
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  #

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model




