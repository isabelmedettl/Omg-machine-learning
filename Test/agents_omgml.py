from rl.agents import DQNAgent  # Need to -> pip install keras-rl2
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Activation
from tensorflow.keras.optimizers import Adam
from gym.wrappers import FrameStack

import environment_omgml


env = environment_omgml.Environment()
#env = FrameStack(env, 3)

states = env.observation_space.shape
actions = env.action_space.n

def build_model(states, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=states))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)

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
dqn.compile(Adam(lr=1e-4))

dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)
dqn.save_weights('SavedWeights/1k-test/dqn_weights.h5')