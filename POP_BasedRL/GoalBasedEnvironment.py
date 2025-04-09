import json
import gym
import numpy as np
import os
import copy
from collections import defaultdict
from gym import spaces
from GPTFeedbackConnector import GPTFeedbackConnector


class GoalBasedEnvironment(gym.Env):
    def __init__(self, env_config, file_path):
        super(GoalBasedEnvironment, self).__init__()

        # Load configuration from JSON
        with open(file_path, 'r') as file:
            config = json.load(file)

        # Extract goal from filename
        self.goal = os.path.splitext(os.path.basename(file_path))[0]

        self.actions = config.get("steps", {})
        if env_config['constraints_flag']=="LLM": # an option to initialize constrains by LLM
            self.edges = config.get("constraints_LLM", [])['constraints']
        elif env_config['constraints_flag']==True: #in case of LLM no constrains info used
            self.edges = config.get("edges", [])

        self.end_state = [ii for ii , k in enumerate(self.actions.values()) if k == 'END'][0]

        self.action_space = spaces.Discrete(len(self.actions))  # Number of available actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.actions),), dtype=np.float32)

        self.state = 0  # Initial state representation
        self.steps_taken = 0
        self.max_steps = len(self.actions)
        self.constraints_flag = env_config['constraints_flag']
        self.reward_type = env_config['reward_type']

        if env_config['constraints_flag']=="LLM" or env_config['constraints_flag']==True:
            self.valid_transitions = {src: [] for src in range(self.max_steps - 1)}
            for src, dest in self.edges:
                self.valid_transitions[src].append(dest)
            self.update_valid_transitions = self.update_valid_transitions_func()
            self.preconditions = self.extract_preconditions_from_transitions()

        else:
            self.valid_transitions = []
            self.update_valid_transitions = []

        self.current_step = "0"  # Start from step "0"

    def step(self, action):
        """Take an action in the environment and return state, reward, done, and info."""
        reward = 0.0
        done = False
        info = {}

        # Track the number of steps
        self.steps_taken += 1

        # Semantic preconditions: what must be visited before a certain action
        # preconditions = {
        #     7: {5, 9, 12},  # blitz → needs banana, egg, flour
        #     11: {7},  # pour → needs blitz
        #     6: {11},  # cook → needs pour
        #     8: {6},  # flip → needs cook
        #     10: {8},  # cook more → needs flip
        #     2: {14},  # serve → needs transfer
        #     15: {2}  # END → needs serve
        # }

        # Check for repeated actions
        if action in self.visited_actions:
            reward += -10.0
            done = True
            info["reason"] = "repeated action"

        # Step limit reached
        elif self.steps_taken >= self.max_steps:
            reward += -5.0
            done = True
            info["reason"] = "step limit reached"

        # Premature END
        elif action == self.end_state and len(self.visited_actions) < len(self.actions) - 1:
            if action not in self.update_valid_transitions[self.state]:
                reward += -10.0
                info["reason"] = "premature END and invalid transition"
            else:
                reward += 1.0
                info["reason"] = "premature END but valid transition"
            done = True

        # Successful completion
        elif action == self.end_state and len(self.visited_actions) == len(self.actions) - 1:
            reward += 10.0
            done = True
            info["reason"] = "successful completion"

        # Normal transitions
        elif not self.state == self.end_state:
            if action not in self.update_valid_transitions[self.state]:
                reward += -5.0
                done = True
                info["reason"] = "invalid transition"
            else:
                #Check semantic preconditions
                required = self.preconditions.get(action, set())
                if not required.issubset(self.visited_actions):
                    reward += -5.0
                    info["reason"] = f"violated preconditions for action {action}"
                    info["missing"] = list(required - self.visited_actions)
                    done = True
                else:
                    reward += 5.0
                    info["reason"] = "valid logical transition"

        # Advance state
        self.state = action
        self.current_step = action
        self.visited_actions.append(action)

        return self.state, reward, done, info

    def step_LLM(self, action):
        reward = 0
        done = False

        # Track the number of steps
        self.steps_taken += 1

        # Step limit reached
        if self.steps_taken >= self.max_steps:
            reward += -5.0
            done = True


        if action == self.end_state:
            done = True

        # Valid progress
        self.state = action
        self.current_step = action
        self.visited_actions.append(action)

        return self.state, reward, done, {}

    def step_MCC(self, action):
        """Take an action in the environment and return state, reward, done, and info."""
        reward = 0
        done = False

        # Track the number of steps
        self.steps_taken += 1

        # Check for repeated actions
        if action in self.visited_actions:
            reward += -5.0
            done = True

        # Step limit reached
        elif self.steps_taken >= self.max_steps:
            reward += -5.0
            done = True

        # Check for premature END

        elif action == self.end_state and len(self.visited_actions) < len(self.actions) - 1:
            if not self.update_valid_transitions == []:
                if not action in self.update_valid_transitions[self.state]:
                    reward += -10.0
                else:
                    reward += 1.0
                done = True

        # Check for successful completion
        elif action == self.end_state and len(self.visited_actions) == len(self.actions) - 1:
            reward += 10.0
            done = True

        # Check for invalid transition
        elif not self.state == self.end_state:
            if not self.update_valid_transitions == []:
                if action not in self.update_valid_transitions[self.state]:
                    reward += -1.0
                    done = True
                else:
                    reward += 1.0

        # Valid progress
        self.state = action
        self.current_step = action
        self.visited_actions.append(action)

        return self.state, reward, done, {}

    def update_valid_transitions_func(self):
        """
        Update the valid_transitions dictionary by adding interconnections for states with multiple transitions.
        """

        valid_transitions = copy.deepcopy(self.valid_transitions)
        valid_transitions_updated = copy.deepcopy(valid_transitions)
        converted_dict = {int(k): v for k, v in self.actions.items()}
        for ii in converted_dict.keys():
            if ii not in valid_transitions_updated.keys() and ii!=self.end_state:
                valid_transitions_updated[ii] = []


        # First, identify states with multiple transitions
        for state, transitions in valid_transitions.items():
            if len(transitions) > 1:
                # For each state in the transition set, add connections to all others
                for t in transitions:
                    if t not in valid_transitions:
                        valid_transitions_updated[t] = []
                    # Ensure all other transitions are connected
                    for other in transitions:
                        if other != t and other not in valid_transitions[t]:
                            valid_transitions_updated[t].append(other)

        return valid_transitions_updated

    def extract_preconditions_from_transitions(self):
        preconditions = defaultdict(set)
        for src, dest_list in self.valid_transitions.items():
            for dest in dest_list:
                preconditions[dest].add(src)
        return dict(preconditions)


    def step_TD(self, action):
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
        self.state = 0 #np.random.choice(range(len(self.actions)-1))
        self.current_step = 0#self.state.copy()
        self.visited_actions = [0]
        return self.current_step

    def compute_reward(self, episode):
        """Sparse reward applied only when reaching the END state."""

        connector = GPTFeedbackConnector()
        action_sequence = [episode_i[0] for episode_i in episode]
        action_sequence.append(episode[-1][1])
        episode_reward = connector.evaluate_sequence(action_sequence= action_sequence, actions = self.actions, goal = self.goal)
        episode_copy = episode.copy()

        bad_transitions = episode_reward['bad transitions']
        if episode_reward['reward']==0:
            sparse_reward = -10
        else:
            sparse_reward = 10


        if episode_reward['reward']==1 and bad_transitions==[]:
            good_transitions = [[i[0],i[1]] for i in episode]
        else:
            good_transitions = episode_reward['good transitions']
        self.valid_transitions = {i[0]:[i[1]] for i in good_transitions}
        self.update_valid_transitions = self.update_valid_transitions_func()
        bad_transitions_filtered = self.filter_transitions_by_sequence(bad_transitions, action_sequence)
        filtered_bad_tr = [tr for tr in bad_transitions_filtered if tr not in good_transitions]
        state_transitions = [state_i[0] for state_i in filtered_bad_tr]

        # filter out from bad_transitions the goog_transitions (LLM might wrong
        visited_list = list(self.visited_actions)
        for t in reversed(range(len(episode))):
            reward_t = 0
            visited_actions = []

            visited_actions = visited_list[0:t+1]
            state, action, reward, act_prob = episode[t]

            # Check for repeated actions
            if action in visited_actions:
                reward_t += -5.0
                done = True

            # Step limit reached
            elif t >= self.max_steps:
                reward_t += -5.0
                done = True

            # Check for premature END

            elif action == self.end_state and len(visited_actions) <= len(self.actions):
                if not self.update_valid_transitions == []:
                    if state in self.update_valid_transitions.keys():
                        if not action in self.update_valid_transitions[state]:
                            reward_t += -10.0
                        else:
                            reward_t += 1.0
                        done = True

            # Check for successful completion
            elif action == self.end_state and len(visited_actions) == len(self.actions) - 1:
                reward_t += 10.0
                done = True

            # Check for invalid transition
            elif not state == self.end_state:
                if not self.update_valid_transitions == []:
                    if state in self.update_valid_transitions.keys():
                        if action not in self.update_valid_transitions[state]:
                            reward_t += -1.0
                            done = True
                        else:
                            reward_t += 1.0

            # if t == self.max_steps-1:
            #     if action == self.end_state:
            #         if episode_reward['reward']==1:
            #             reward_t += 10
            #             # at the ebd of the sequence , LLM indicates good sequence, the goal is reached then the reward is high
            #         else:
            #             reward_t += -10
            #
            # else:
            #
            #     if t==len(episode)-1 or action == self.end_state:
            #         reward_t += -10
            #     if action in visited_list[0:t]: # state taken twice , the same action taken twice
            #         reward_t += -1
            #     elif state in state_transitions:
            #         reward_t += -1  # not ligall transition

            episode_copy[t] = (state, action, reward_t, act_prob)

        return episode_copy, sparse_reward

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


