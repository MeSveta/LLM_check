import numpy as np
import random
from GoalBasedEnvironment import GoalBasedEnvironment
from RLAgent import RLAgent
import json
import yaml
import os




def main(config):
    json_dir = config['env']['json_path']
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
            env = GoalBasedEnvironment(env_config = config['env'],file_path = full_path)
            agent = RLAgent(agent_config = config['agent'],init_sequence_path = full_path,
                constraints=env.valid_transitions,
                state_space_size=num_states,
                action_space_size=num_actions
            )

            agent.train_TD(env, num_episodes=5)
            agent.train_MCC(env, num_episodes=5)

            # Print learned policy
            policy = agent.get_policy()
            print("\nLearned Policy:")
            for state, action in policy.items():
                print(f"State {state} -> Action {action} ({steps[str(action)]})")


if __name__ == "__main__":
    with open("POP_RL_config_TD.yaml", "r") as f:
        config = yaml.safe_load(f)

    #file_path = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM\blenderbananapancakes.json"
    config['env']['json_path'] = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM"
    config['agent']['init_sequence_path'] = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM"

    main(config)
