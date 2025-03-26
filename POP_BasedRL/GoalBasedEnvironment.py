import json
import gym
import numpy as np
import os
from gym import spaces
from GPTFeedbackConnector import GPTFeedbackConnector


class GoalBasedEnvironment(gym.Env):
    def __init__(self, env_config, file_path):
        super(GoalBasedEnvironment, self).__init__()

        # Load configuration from JSON
        with open(file_path, 'r') as file:
            config = json.load(file)

        # Extract goal from filename
        self.goal = os.path.splitext(os.path.basename(env_config['json_path']))[0]

        self.actions = config.get("steps", {})
        self.edges = config.get("constraints_LLM", [])['constraints']

        self.end_state = [ii for ii , k in enumerate(self.actions.values()) if k == 'END'][0]

        self.action_space = spaces.Discrete(len(self.actions))  # Number of available actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.actions),), dtype=np.float32)

        self.state = np.zeros(1)  # Initial state representation
        self.steps_taken = 0
        self.max_steps = len(self.actions)
        self.constraints_flag = env_config['constraints_flag']

        if env_config['constraints_flag']:
            self.valid_transitions = {src: [] for src in range(self.max_steps - 1)}
            for src, dest in self.edges:
                self.valid_transitions[src].append(dest)
        else:
            self.valid_transitions = []

        self.current_step = "0"  # Start from step "0"

    def step(self, action):
        """Take an action in the environment."""


        if (action in self.visited_actions or
                (action == self.end_state and len(self.visited_actions) < len(self.actions) - 1)):  # Prevent END early
            reward = -1.0  # Penalty for invalid or repeated action or early END
            done = True
        else:


            self.state = action  # Mark action as taken
            self.visited_actions.add(action)
            self.steps_taken += 1
            self.current_step = action  # Move to next step
            done = self.current_step == self.end_state  # Check if END step is reached
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        """Reset the environment."""
        #self.state = np.zeros(1)
        self.steps_taken = 0
        self.current_step = 0
        self.visited_actions = set([0])
        return self.current_step

    def compute_reward(self, episode):
        """Sparse reward applied only when reaching the END state."""

        connector = GPTFeedbackConnector()
        action_sequence = [episode_i[0] for episode_i in episode]
        action_sequence.append(episode[-1][1])
        episode_reward = connector.evaluate_sequence(action_sequence= action_sequence, actions = self.actions, goal = self.goal)
        episode_copy = episode.copy()

        bad_transitions = episode_reward['bad transitions']
        bad_transitions_filtered = self.filter_transitions_by_sequence(bad_transitions, action_sequence)
        state_transitions = [state_i[0] for state_i in bad_transitions_filtered]

        visited_list = list(self.visited_actions)
        for t in reversed(range(len(episode))):
            reward_t = 0
            state, action, reward, act_prob = episode[t]
            if t == self.max_steps-1:
                if action == self.end_state:
                    if episode_reward['reward']==1:
                        reward_t += 10
                        # at the ebd of the sequence , LLM indicates good sequence, the goal is reached then the reward is high
                    else:
                        reward_t += -10

            else:

                if t==len(episode)-1 or action == self.end_state:
                    reward_t += -10
                if action in visited_list[0:t]: # state taken twice , the same action taken twice
                    reward_t += -1
                elif state in state_transitions:
                    reward_t += -1  # not ligall transition

            episode_copy[t] = (state, action, reward_t, act_prob)

        return episode_copy

    def filter_transitions_by_sequence(self,transitions, sequence):
        """
        Keep only the transitions that appear in order in the sequence.
        Example: [6, 7] must appear as consecutive elements in the sequence.
        """
        sequence = [int(x) for x in sequence]  # In case of np.int64
        valid_pairs = set(zip(sequence, sequence[1:]))

        return [pair for pair in transitions if tuple(pair) in valid_pairs]

    def render(self, mode='human'):
        """Render the current state of the environment."""
        print(f"Goal: {self.goal}")
        print(f"Current Step: {self.actions.get(self.current_step, 'Unknown')}")
        print(f"Actions taken: {self.state}")


