from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation
from tensorflow.keras.optimizers import Adam

def create_dqn_model(input_shape, action_space):
    model = Sequential()  # According to Tensorflow, sequential is only appropriate when the model has ONE input and ONE output, we have many more. Maybe reconsider.
    model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=input_shape))  # Because we use images, we need to first set up a convolutional network and then flatten it down. Input_shape is image
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(action_space))  # Action_space is how many actions we have
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model