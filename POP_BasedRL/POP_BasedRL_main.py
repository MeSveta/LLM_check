import numpy as np
import random
from GoalBasedEnvironment import GoalBasedEnvironment
from RLAgent import RLAgent
import json
import yaml
import os
from utils.generate_results import PlotResults

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, np.int64):
            return int(obj)  # Convert np.int64 to regular int
        return super().default(obj)


def main(config):
    json_dir = config['env']['json_path']
    save_dir = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL"
    num_episodes = 200000
    if os.path.isdir(json_dir):
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                full_path = os.path.join(json_dir, filename)
            config_path = full_path
            with open(config_path, 'r') as file:
                data = json.load(file)
            # Path to your JSON file

            constraints = data.get("constraints_LLM", [])

            steps = data["steps"]
            num_actions = len(steps)
            num_states = num_actions  # Assuming one-to-one mapping of steps to state ids

            # Create the environment
            #env_MCC = GoalBasedEnvironment(env_config = config['env'],file_path = full_path)
            env_TD = GoalBasedEnvironment(env_config=config['env'], file_path=full_path)
            # agent_MCC = RLAgent(agent_config = config['agent'],init_sequence_path = full_path,
            #     constraints=env_MCC.valid_transitions,
            #     state_space_size=num_states,
            #     action_space_size=num_actions,
            #     num_episodes=num_episodes
            # )

            agent_TD = RLAgent(agent_config=config['agent'], init_sequence_path=full_path,
                                constraints=env_TD.valid_transitions,
                                state_space_size=num_states,
                                action_space_size=num_actions,
                                num_episodes=num_episodes
                                )

            #agent_MCC.train_MCC(env_MCC, num_episodes=num_episodes)
            agent_TD.train_TD(env_TD, num_episodes=5)

            res = {}
            res['Q'] = agent_TD.Q
            res['target_policy'] = agent_TD.target_policy
            res['rewards_hist'] = agent_TD.reward_hist
            res['env_constrains'] = agent_TD.valid_transitions
            res['res_constrains_updated'] = agent_TD.update_valid_transitions
            res['goal'] = agent_TD.goal
            res['steps'] = agent_TD.actions

            file_name = 'TD_agent' + agent_TD.goal + '.json'
            with open(file_name, "w") as f:
                json.dump(res, f, indent=4, cls=CustomEncoder)


            # Print learned policy
            policy = agent_TD.generate_target_policy()
            policy = agent_TD.target_policy
            print("\nLearned Policy:")
            state_u = 0
            for state, action in policy.items():
                state = state_u
                action = policy[state]
                print(f"State {state} -> Action {action} ({steps[str(action)]})")
                state_u = action

            #prepare for plot
            rewards = [agent_TD.reward_hist]
            gen_res = PlotResults(env = env_TD, Q = agent_TD.Q, rewards = rewards, save_dir = save_dir)
            gen_res.plot_rewards()




if __name__ == "__main__":
    with open("POP_RL_config_TD.yaml", "r") as f:
        config = yaml.safe_load(f)

    #file_path = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM\blenderbananapancakes.json"
    # config['env']['json_path'] = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM"
    # config['agent']['init_sequence_path'] = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM"

    main(config)
