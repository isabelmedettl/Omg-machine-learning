
from rl.agents import DQNAgent  # Need to -> pip install keras-rl2
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation
from keras.optimizers import Adam
import gym

import environment_omgml


env = environment_omgml.Environment()
height, width, channels = env.observation_space.shape
actions = env.action_space

def build_model(height, width, channels, action_space):
    model = Sequential()  # According to Tensorflow, sequential is only appropriate when the model has ONE input and ONE output, we have many more. Maybe reconsider.
    model.add(Conv2D(16, (8, 8), strides=(4, 4), input_shape=(18, height, width, channels)))  # Because we use images, we need to first set up a convolutional network and then flatten it down. Input_shape is image
    model.add(Activation('relu'))
    model.add(Conv2D(32, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(action_space))  # Action_space is how many actions we have
    model.compile(loss='mse', optimizer=Adam(lr=0.005))
    return model


model = build_model(height, width, channels, actions)

model.summary()

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=1000)
    memory = SequentialMemory(limit=100, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=100
                   )
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=5*1e-4))

dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)
dqn.save_weights('SavedWeights/1k-test/dqn_weights.h5')