from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation
from tensorflow.keras.optimizers import Adam

def create_dqn_model(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(action_space))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model