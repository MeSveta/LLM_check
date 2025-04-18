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

    def create_behavior_policy(self, state, behavior_policy_epsilon):
        """Create an ε-soft behavior policy for exploration."""

        rand_val = np.random.rand()
        greedy_act = self.target_policy[state]

        if rand_val > behavior_policy_epsilon:
            return greedy_act, (1 - behavior_policy_epsilon + behavior_policy_epsilon / self.action_space_size)
        else:
            action = random.randint(0, self.action_space_size - 1)
            if action == greedy_act:
                return action, (1 - behavior_policy_epsilon + behavior_policy_epsilon / self.action_space_size)
            else:
                return action, behavior_policy_epsilon / self.action_space_size

            # """Choose a random valid action from the constraint map."""
            # valid_actions = self.constraints.get(state, list(range(self.action_space_size)))
            # if valid_actions:
            #     action = random.choice(valid_actions)
            # else:
            #     action = random.randint(0, self.action_space_size - 1)


    def generate_episode(self, env):
        """Generates an episode using the behavior policy."""
        episode = []
        episodes = []
        episode_copy =  []

        done = False


        if self.agent_config['train']['mode'] == 'MCC' or self.agent_config['train']['mode'] == 'Sarsa':
            sparse_reward = []
            for batch_i in range(self.agent_config['train']['batch_size']):
                episode = []
                state = env.reset()
                done = False
                while not done:
                    #print(state)
                    action, act_prob = self.create_behavior_policy(state, self.behavior_policy_epsilon)

                    if env.reward_type=='LLM':
                        next_state, reward, done, _ = env.step_LLM(action)
                    else:
                        next_state, reward, done, _ = env.step_MCC(action)

                    episode.append((state, action, reward, act_prob))
                    state = next_state
                episodes.append(episode)

            if done and env.reward_type=='LLM':
                # send the sequence to LLM to check the final sequence and sub transitions
                episode_copy, sparse_reward = env.compute_reward(episodes)
                episode = episode_copy
        elif self.agent_config['train']['mode'] == 'TD':
                action, act_prob = self.create_behavior_policy(state)
                next_state, reward, done, _ = env.step_TD(action)
                episode.append((state, action, reward, act_prob))
                state = next_state

        return episode,episodes,sparse_reward

    from collections import defaultdict
    import numpy as np

    # === Monte Carlo Batch Q-Value Update Function ===
    def MCC_batch_update_Q(self,episodes_LLM, episodes_RL, gamma=0.99, success_weight=1.0, fail_weight=0.3):

        """Per-transition frequency-aware MC update.
        Transitions that appear mostly in good sequences get higher weight.
        """
        transition_counts = defaultdict(lambda: {"good": 0, "bad": 0})
        q_returns = defaultdict(list)

        # First pass: count transition frequency
        for ii, trajectory in enumerate(episodes_RL):
            if len(episodes_RL)!=len(episodes_LLM):
                continue
            if episodes_LLM[ii]['reward']>0:
                reward = 10
            else:
                reward = -10

            tag = "good" if reward > 0 else "bad"
            for t in range(len(trajectory)):
                s, a, rr, act_prob = trajectory[t]
                transition_counts[(s, a)][tag] += 1

        # Second pass: collect weighted returns
        for ii, trajectory in enumerate(episodes_RL):
            if len(episodes_RL)!=len(episodes_LLM):
                continue
            if episodes_LLM[ii]['reward'] > 0:
                reward = 10
            else:
                reward = -10
            T = len(trajectory)
            for t in range(len(trajectory)):
                s, a, rr, act_prob = trajectory[t]
                counts = transition_counts[(s, a)]
                total = counts["good"] + counts["bad"]

                # Weight for this instance based on frequency of success/fail
                if reward > 0:
                    weight = (counts["good"] / total) * success_weight
                else:
                    weight = (counts["bad"] / total) * fail_weight

                G_t = reward * (gamma ** (T - t - 1))
                q_returns[(s, a)].append((G_t, weight))

        # Final Q-value update
        for (s, a), returns in q_returns.items():
            total_weight = sum(w for _, w in returns)
            if total_weight == 0:
                continue
            weighted_sum = sum(g * w for g, w in returns)
            avg_G = weighted_sum / total_weight
            self.Q[s][a] += self.alpha * (avg_G -self.Q[s] [a])



    def MCC_update(self,episode,W,G):
        for t in reversed(range(len(episode))):
            state, action, reward, act_prob = episode[t]
            G = self.gamma * G + reward  # Compute return

            # Update cumulative sum of weights
            self.C[state][action] += W

            # Update action-value function using weighted importance sampling
            #self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
            self.Q[state][action] += self.alpha* (G - self.Q[state][action])

            # Improve policy (greedy update)
            self.target_policy[state] = self.max_argmax(self.Q[state])

            # Check if action differs from target policy
            if action != self.target_policy[state]:
                break  # Stop the update early

            # Update importance sampling weight
            #W *= 1 / act_prob
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

    def train_LLM(self, env, full_path, num_episodes=10000):
        """Trains the agent using Monte Carlo with importance sampling."""
        for episode_index in range(num_episodes):
            # if episode_index>10:
            #     self.behavior_policy_epsilon = 0.4
            # else:
            self.behavior_policy_epsilon = max(0.01, 0.3 * 0.9)
            episode_LLM, episodes_RL, sparse_reward = self.generate_episode(env)
            G = sparse_reward
            W = 1

            #self.reward_hist[episode_index] = np.sum([ii[2] for ii in episode])
            if self.agent_config['train']['mode'] == 'MCC':
                if len(episodes_RL)>0:
                    self.MCC_batch_update_Q(episodes_LLM = episode_LLM,episodes_RL=episodes_RL)
                else:
                    W,G = self.MCC_update(episodes_RL,W,G)
            if self.agent_config['train']['mode'] == 'Sarsa':
                for t in episodes_RL:
                    self.update(t)

            if len(episodes_RL)>1:
                self.reward_hist[episode_index] += 0
                for ii, trajectory in enumerate(episodes_RL):
                    if len(episodes_RL) != len(episode_LLM):
                        continue
                    if episode_LLM[ii]['reward'] > 0:
                        reward = 1
                    else:
                        reward = -1
                    self.reward_hist[episode_index] += reward
                print(f'Episode: {episode_index}: reward: {self.reward_hist[episode_index]}')
                policy = self.generate_target_policy()
                # print("\nLearned Policy:")
                # state_u = 0
                # for state, action in self.policy.items():
                #     state = state_u
                # action = policy[state]
                # print(f"State {state} -> Action {action} ({env.actions[str(action)]})")
                # state_u = action
                # if state_u == env.end_state:
                #     break
            else:
                action_sequence = [episode_i[0] for episode_i in episodes_RL]
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
                    action_next, dontcare = self.create_behavior_policy(next_state,self.behavior_policy_epsilon)
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

            action, act_prob = self.create_behavior_policy(state,self.behavior_policy_epsilon)
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
    def get_action(self, state):
        if np.random.rand() < self.behavior_policy_epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return self.max_argmax(self.Q[state])

    def update(self, s, a, r, s_, a_):
        target = r if a_ == 0 and r <= 0 else r + self.gamma * self.Q[s_][a_]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    # === RL-Compatible step function ===
    def rl_step(self,env, action):
        reward = 0.0
        done = False
        info = {}

        env.steps_taken += 1

        if action in env.visited_actions:
            reward = -1.0
            done = True
            info["reason"] = "repeated action"

        elif env.steps_taken >= env.max_steps:
            reward = -5.0
            done = True
            info["reason"] = "step limit reached"

        elif action == env.end_state and len(env.visited_actions) < len(env.actions) - 1:
            reward = -10.0
            done = True
            info["reason"] = "premature END"

        elif action == env.end_state and len(env.visited_actions) == len(env.actions) - 1:
            reward = 10.0
            done = True
            info["reason"] = "successful completion"

        elif not env.state == env.end_state:
            if action not in env.update_valid_transitions.get(env.state, []):
                reward = -5.0
                done = True
                info["reason"] = "invalid transition"
            else:
                required = env.preconditions.get(action, set())
                if not required.issubset(env.visited_actions):
                    reward = -5.0
                    done = True
                    info["reason"] = f"violated preconditions for action {action}"
                    info["missing"] = list(required - set(env.visited_actions))
                else:
                    reward = 6.0
                    info["reason"] = "valid logical transition"

        env.state = action
        env.current_step = action
        env.visited_actions.append(action)

        return env.state, reward, done, info

    def train_Sarsa(self, env, n=1, num_episodes=1000):
        rewards = []
        reason_log = []
        success_count = 0

        for ep in range(num_episodes):
            self.behavior_policy_epsilon = max(0.01, self.behavior_policy_epsilon * 0.9)
            episode_trace = []
            state = env.reset()
            # env.visited_actions.append(0)
            action = self.get_action(state)
            ep_reward = 0
            done = False

            while not done:
                next_state, reward, done, info = self.rl_step(env, action)
                next_action = self.get_action(next_state)
                # Skip update if repeated action to avoid learning it as good
                if not done or (info.get("reason") != "repeated action" and reward > 0):
                    self.update(state, action, reward, next_state, next_action)
                if done:  # and not (info.get("reason") != "repeated action" and reward > 0):
                    self.update(state, action, reward, next_state, 0)

                state, action = next_state, next_action
                ep_reward += reward
                episode_trace.append((state, action, reward, info))

            rewards.append(ep_reward)
            episode_reasons = [info.get("reason", "") for (_, _, _, info) in episode_trace if isinstance(info, dict)]
            reason_log.extend(episode_reasons)

            if "successful completion" in episode_reasons:
                success_count += 1

            print(f"Episode {ep} | Reward: {ep_reward:.2f} | Trace: {episode_trace} | Reasons: {episode_reasons}")

        self.print_policy()
        self.rewards = rewards



