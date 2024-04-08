#from rl.agents import DQNAgent  # Need to -> pip install keras-rl2
#from rl.memory import SequentialMemory
#from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import keras
from keras import layers
from gymnasium.wrappers import FrameStack
import numpy as np
import tensorflow as tf
import environment_omgml
from datetime import datetime

env = environment_omgml.Environment()
#env = FrameStack(env, 3)

states = env.observation_space.shape
actions = env.action_space

num_actions = 18

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for pasta rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 8  # Size of batch taken from replay buffer
max_steps_per_episode = 1000
max_episodes = 5  # Limit training episodes, will run until solved if smaller than 1

def create_q_model():
    # Network defined by the Deepmind paper
    return keras.Sequential(
        [
            # Convolutions on the frames on the screen
            layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=states),
            layers.Conv2D(64, 4, strides=2, activation="relu"),
            layers.Conv2D(64, 3, strides=1, activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_actions, activation="linear"),
        ]
    )




#model = create_q_model()

# The first model makes the predictions for Q-values which are used to
# make a action.
print(states)
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()
model.summary()


# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
episode_reward_history_max = 100
running_reward = 0
running_reward_max = 100
episode_count = 0
frame_count = 0
step_counter = 0
# Number of frames to take random action and observe output
epsilon_random_steps = 50
# Number of frames for exploration
epsilon_greedy_steps = 100
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000
# Train the model after 4 actions
''' this could be wrong, check if statement in loop'''
update_after_actions = 1
# How often to update the target network
''' what is this, is it how often conv.updates are done?? '''
update_target_network = 100
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:
    observation, _ = env.reset()
    state = np.array(observation)
    #state = observation
    episode_reward = 0

# define max_steps_per_episode
    for timestep in range(1, max_steps_per_episode):
        #frame_count += 1
        step_counter += 1

        # Use epsilon-greedy for exploration
        if step_counter < epsilon_random_steps or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = keras.ops.convert_to_tensor(state)
            state_tensor = keras.ops.expand_dims(state_tensor, 0) #adding another dimension to the tensor
            action_probabilities = model(state_tensor, training=False)
            # Take best action
            action = keras.ops.argmax(action_probabilities[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_steps
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward
        print("Episode reward: ", int(episode_reward))

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)

        '''
        this might be a huge issue, what is done? Should done be frame-render-done, game-is-done or step-is-done?
        '''
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if step_counter % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = keras.ops.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * keras.ops.amax(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = keras.ops.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step_counter % update_target_network == 0:
            # update the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, step_counter))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > episode_reward_history_max:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > running_reward_max:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        model.save(f'model_from{datetime.now()}')
        break

    if (
        max_episodes > 0 and episode_count >= max_episodes
    ):  # Maximum number of episodes reached
        print("Stopped at episode {}!".format(episode_count))
        model.save(f'model_from{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')
        break



# OLD STUFF
'''
env = environment_omgml.Environment()
env = FrameStack(env, 3)

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

'''