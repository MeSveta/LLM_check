import numpy as np
import random
from collections import defaultdict
import json


class RLAgent:
    def __init__(self,agent_config, init_sequence_path, constraints, state_space_size, action_space_size, alpha=0.1, gamma=0.9, behavior_policy_epsilon=0.2, num_episodes=10000):
        self.agent_config = agent_config
        self.init_sequence_path = init_sequence_path
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.Q_init_flag = agent_config['Q_init']
        self.Q = np.zeros((self.state_space_size - 1, self.action_space_size))
        self.C = np.zeros((self.state_space_size - 1, self.action_space_size))
        self.target_policy = self.generate_target_policy()
        self.init_sequence = self.extract_init_sequence()
        # cumulative weights for IS # Q[state][action]
        self.returns = defaultdict(list)  # For Monte Carlo returns
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.behavior_policy_epsilon = behavior_policy_epsilon
        self.Q_init()  # Q[state, action]
        self.G = 0  # return
        self.W = 1  # importance sampling weight
        self.constraints = constraints
        self.reward_hist = np.zeros(shape=(num_episodes), dtype=np.float32)

    def max_argmax(self, input):
        best_action = np.random.choice([i for i, value in enumerate(input) if value == max(input)])
        return best_action

    def generate_target_policy(self):
        """Returns the greedy policy based on Q-values."""
        return {state: self.max_argmax(self.Q[state]) for state in range(self.state_space_size-1)}


    def _build_constraint_map(self, constraints):
            """Converts constraints list into a dict of valid actions per state."""
            # constraint_map = defaultdict(list)
            # for state, action in constraints:
            #     constraint_map[state].append(action)
            return constraints

    def create_behavior_policy(self, state):
        """Create an ε-soft behavior policy for exploration."""

        rand_val = np.random.rand()
        greedy_act = self.target_policy[state]

        if rand_val > self.behavior_policy_epsilon:
            return greedy_act, (1 - self.behavior_policy_epsilon + self.behavior_policy_epsilon / self.action_space_size)
        else:
            action = random.randint(0, self.action_space_size - 1)
            if action == greedy_act:
                return action, (1 - self.behavior_policy_epsilon + self.behavior_policy_epsilon / self.action_space_size)
            else:
                return action, self.behavior_policy_epsilon / self.action_space_size

            # """Choose a random valid action from the constraint map."""
            # valid_actions = self.constraints.get(state, list(range(self.action_space_size)))
            # if valid_actions:
            #     action = random.choice(valid_actions)
            # else:
            #     action = random.randint(0, self.action_space_size - 1)


    def generate_episode(self, env):
        """Generates an episode using the behavior policy."""
        episode = []
        episode_copy =  []
        state = env.reset()
        done = False


        if self.agent_config['train']['mode'] == 'MCC':
            while not done:
                print(state)
                action, act_prob = self.create_behavior_policy(state)

                if env.reward_type=='LLM':
                    next_state, reward, done, _ = env.step_LLM(action)
                else:
                    next_state, reward, done, _ = env.step_MCC(action)

                episode.append((state, action, reward, act_prob))
                state = next_state

            if done and env.reward_type=='LLM':
                # send the sequence to LLM to check the final sequence and sub transitions
                episode_copy = env.compute_reward(episode)
                episode = episode_copy
        elif self.agent_config['train']['mode'] == 'TD':
                action, act_prob = self.create_behavior_policy(state)
                next_state, reward, done, _ = env.step_TD(action)
                episode.append((state, action, reward, act_prob))
                state = next_state


        return episode

    def MCC_update(self,episode,W,G):
        for t in reversed(range(len(episode))):
            state, action, reward, act_prob = episode[t]
            G = self.gamma * G + reward  # Compute return

            # Update cumulative sum of weights
            self.C[state][action] += W

            # Update action-value function using weighted importance sampling
            self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

            # Improve policy (greedy update)
            self.target_policy[state] = self.max_argmax(self.Q[state])

            # Check if action differs from target policy
            if action != self.target_policy[state]:
                break  # Stop the update early

            # Update importance sampling weight
            W *= 1 / act_prob
            if W == 0:
                break  # Stop early if weight goes to zero
        return W,G

    def extract_init_sequence(self):
        # Load configuration from JSON
        if self.Q_init_flag == 'labeled_sequence':
            with open(self.init_sequence_path, 'r') as file:
                config = json.load(file)
            return config['valid_sequence']
        else:
            return []

    def initialize_with_episode(self, sequence, reward):
        """
        Manually initialize Q, C, and target_policy using a known correct sequence.
        :param sequence: List of action indices in correct order.
        :param reward: Final reward at the end of the sequence.
        """
        G = reward
        for t in reversed(range(len(sequence))):
            state = sequence[t - 1] if t > 0 else 0  # initial state is 0
            action = sequence[t]
            self.C[state][action] += 1
            self.Q[state][action] += (G - self.Q[state][action]) / self.C[state][action]
            self.target_policy[state] = action
            G = self.gamma * G  # decay reward for earlier steps

    def Q_init(self):

        if self.Q_init_flag=='labeled_sequence':
            sequence = self.init_sequence
            episode = []
            episode = [
                (sequence[i], sequence[i + 1], 1.0, 1.0)
                for i in range(len(sequence) - 1)]
            W =1
            G = 0
            W, G = self.MCC_update(episode, W, G)

    def train_MCC(self, env, num_episodes=10000):
        """Trains the agent using Monte Carlo with importance sampling."""
        for episode_index in range(num_episodes):
            episode = self.generate_episode(env)
            G = 0
            W = 1

            self.reward_hist[episode_index] = np.sum([ii[2] for ii in episode])
            W,G = self.MCC_update(episode,W,G)

            action_sequence = [episode_i[0] for episode_i in episode]
            if np.mod(episode_index, 1) == 0:
                print(
                    f'Episode: {episode_index}: {action_sequence}, reward: {self.reward_hist[episode_index]}, epsilon:{self.behavior_policy_epsilon}')

    def train_TD(self, env, num_episodes=1000):
        """Trains the agent using Monte Carlo with importance sampling."""
        for episode_index in range(num_episodes):
            done = False
            episode = []
            state = env.reset()
            G=0
            W=1
            while not done:
                if state!=env.state:
                    y=1
                action, act_prob = self.create_behavior_policy(state)
                if action == env.end_state:
                     y = 1
                next_state, reward, done, _ = env.step(action)


                if done:
                    target = reward
                else:
                    action_next, dontcare = self.create_behavior_policy(next_state)
                    target = reward + self.gamma * self.Q[next_state][action_next]

                self.Q[state][action] += self.alpha * (target-self.Q[state][action])

                # Improve policy (greedy update)
                self.target_policy[state] = self.max_argmax(self.Q[state])
                episode.append((state, action, reward, act_prob))
                state = next_state

            self.reward_hist[episode_index] = np.sum([ii[2] for ii in episode])
            action_sequence = [episode_i[0] for episode_i in episode]
            if np.mod(episode_index, 1) == 0:
                print(
                    f'Episode: {episode_index}: {action_sequence}, reward: {self.reward_hist[episode_index]}, epsilon:{self.behavior_policy_epsilon}')

    def train_nSarsa(self, env, n=1, num_episodes=1000):
        for episode_index in range(num_episodes):
            state = env.reset()
            decay_rate = 0.01

            # Epsilon decay every 1000 episodes
            # if episode_index % 1000 == 0:
            #self.behavior_policy_epsilon = max(0.1, 1 * np.exp(-decay_rate * episode_index))

            action, act_prob = self.create_behavior_policy(state)
            t = 0
            T = float("inf")
            buffer = []
            G = 0.0

            self.reward_hist[episode_index] = 0.0

            while True:
                # Step in environment
                if t < T:
                    next_state, reward, done, _ = env.step(action)
                    self.reward_hist[episode_index] += reward

                    if done:
                        T = t + 1
                        next_action, next_act_prob = None, 1.0
                    else:
                        next_action, next_act_prob = self.create_behavior_policy(next_state)

                    buffer.append((state, action, reward, act_prob))

                tau = t - n + 1
                if tau >= 0:

                    W = 1.0

                    for i in range(tau, min(tau + n, T)):
                        _, _, r_i, _ = buffer[i]
                        G += (self.gamma ** (i - tau)) * r_i

                    # Safe bootstrap
                    if tau + n < T and tau + n < len(buffer):
                        s_n, a_n, _, act_prob_n = buffer[tau + n]
                        G += (self.gamma ** n) * self.Q[s_n][a_n]
                        if self.target_policy[s_n] != a_n:
                            W = 0.0
                        else:
                            W *= 1 / act_prob_n

                    # Final update (even without bootstrapping)
                    if tau < len(buffer):  #
                        s_tau, a_tau, _, _ = buffer[tau]
                        self.Q[s_tau][a_tau] += self.alpha * W * (G - self.Q[s_tau][a_tau])
                        self.target_policy[s_tau] = self.max_argmax(self.Q[s_tau])

                    if tau == T - 1:
                        break


                state = next_state
                env.state = state
                action = next_action
                act_prob = next_act_prob
                t += 1

            # Optional: monitor progress
            if episode_index % 100 == 0:
                print(f"Episode {episode_index}, Total Reward: {self.reward_hist[episode_index]:.2f}, Epsilon: {self.behavior_policy_epsilon:.4f}")

