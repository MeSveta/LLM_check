import numpy as np
from openai import OpenAI
from itertools import permutations
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle

import re
import yaml

def generate_graph(steps,adj_matrix,output_folder_test, title):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    for index, label in steps.items():
        G.add_node(int(index), label=label)

    # Add edges based on adjacency matrix
    for i, row in enumerate(adj_matrix):
        for j, val in enumerate(row):
            if val == 1:
                G.add_edge(i, j)

    # Draw the graph
    pos = {}
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    for i, node in enumerate(nodes):
        pos[node] = (i / (num_nodes - 1), 0)  # x-coordinate varies, y-coordinate is 0

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # or nx.circular_layout(G)
    # Draw graph with labels in boxes
    node_labels = nx.get_node_attributes(G, "label")
    # Draw nodes
    fig, ax = plt.subplots(figsize=(16, 8))
    # Draw edges with parallel curves for multiple edges
    edge_colors = []
    for (u, v, data) in G.edges(data=True):
        if u != v:  # Exclude self-loops for this example
            edge_colors.append('gray')

    # Count parallel edges
    edge_counts = {}
    for (u, v) in G.edges():
        if u != v:  # Exclude self-loops
            edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
            edge_counts[(v, u)] = edge_counts.get((v, u), 0) + 1

    # Custom connection style
    def custom_connection_style(pos_src, pos_dst, num_edges, index):
        mid = (pos_src + pos_dst) / 2
        diff = pos_dst - pos_src
        normal = np.array([-diff[1], diff[0]])
        normal /= np.linalg.norm(normal)

        if num_edges % 2 == 0:
            offset = (index - num_edges / 2 + 0.5) * 0.2
        else:
            offset = (index - (num_edges - 1) / 2) * 0.2

        mid += normal * offset
        return f"arc3,rad={0.2 * offset}"

    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        if u != v:  # Exclude self-loops
            num_edges = edge_counts[(u, v)]
            index = i % num_edges
            connection_style = custom_connection_style(np.array(pos[u]), np.array(pos[v]), num_edges, index)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color='gray',
                                   connectionstyle='arc3,rad=0.2', arrows=True, arrowsize=35)

    # Calculate text sizes and create boxes
    for node, (x, y) in pos.items():
        label = node_labels[node]
        bbox = dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5')
        text = ax.text(x, y, label, ha='center', va='center', bbox=bbox)

        # Get the bounding box of the text
        bbox = text.get_bbox_patch()
        width = bbox.get_width()
        height = bbox.get_height()

        # Create a rectangle patch
        rect = Rectangle((x - width / 2, y - height / 2), width, height,
                         facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)

        # Add text on top of rectangle
        ax.text(x, y, label, ha='center', va='center')

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.axis('off')
    plt.tight_layout()
    title_plot = os.path.join(os.path.join(output_folder_test, 'plot'),title)
    plt.savefig(title_plot+'.jpg',bbox_inches='tight', dpi=300)
    plt.show()

def split_str_convert_to_num(response_str):
    # Split the string into a list of strings
    separated_values = response_str.split(",")

    # Convert each string into an integer
    numeric_values = [int(value) for value in separated_values]

    return(numeric_values)
# Initialize the client
#API_KEY = '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'
API_KEY = 'sk-proj-sbeE4x0JtQjYbNzvzU3ET3BlbkFJY2hPpO3T4X8WWUc61a47'#'sk-proj-sbeE4x0JtQjYbNzvzU3ET3BlbkFJY2hPpO3T4X8WWUc61a47'#'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'
client = OpenAI(api_key=API_KEY)


# Create a chat completion request
#"generate a logical sequence of these actions, the output is only the actions that mentioned, dont add any other words: pour milk, whisk mixture, pour egg, dip bread in mixture, melt butter, put bread in pan, add vanilla extract, flip bread, remove bread from pan, top toast"
#give me the next action after pour milk, choose it from here: (pour egg, whisk mixture, dip bread in mixture, melt butter, put bread in pan, add vanilla extract, flip bread, remove bread from pan, top toast).Dont add any other words."}
#(1.pour milk ,2.pour egg, 3.whisk mixture, 4.dip bread in mixture, 5.melt butter, 6.put bread in pan, 7.add vanilla extract, 8.flip bread, 9.remove bread from pan)"}



# Define the steps and edges
data = {
    "steps": {
        "0": "START",
        "1": "Add-1/2 tsp baking powder to a blender",
        "2": "Serve-Serve the pancakes with chopped strawberries",
        "3": "Melt-Melt a small knob of butter in a non-stick frying pan over low-medium heat",
        "4": "splash-splash maple syrup on plate",
        "5": "Add-Add 1 banana to a blender",
        "6": "Cook-Cook for 1 min or until the tops start to bubble",
        "7": "blitz-blitz the blender for 20 seconds",
        "8": "Flip-Flip the pancakes with a fork or a fish slice spatula",
        "9": "Add-1 egg to a blender",
        "10": "cook-cook for 20-30 seconds more",
        "11": "Pour-Pour three little puddles straight from the blender into the frying pan",
        "12": "Add-1 heaped tbsp flour to a blender",
        "13": "Chop-Chop 1 strawberry",
        "14": "Transfer-Transfer to a plate",
        "15": "END",
    },
    "edges": [
        [14, 2],
        [7, 3],
        [13, 2],
        [2, 4],
        [11, 6],
        [5, 7],
        [12, 7],
        [1, 7],
        [9, 7],
        [6, 8],
        [8, 10],
        [3, 11],
        [10, 14],
        [0, 1],
        [0, 5],
        [0, 9],
        [0, 12],
        [0, 13],
        [4, 15]
    ]
}

import os
import json

# Folder containing the JSON files
folder_path = r"C:\Users\Sveta\PycharmProjects\data\Cook"
config_path = r"C:\Users\Sveta\PycharmProjects\LLM_check\configs\CaptainCook4D"
output_folder_test =  r"C:\Users\Sveta\PycharmProjects\LLM_check\Cook_test_Feb"


# Dictionary to store the data
data = {}
test_vec = []
test_clean_vec= []
test_response_vec = []
GT_vec = []

summary_dict= {"file_name":[],
               "accuracy_test":[],
               "accuracy_test_clean":[],
               "accuracy_response":[],
               "f1_test":[],
               "f1_test_clean":[],
               "f1_response":[]

}

total_summary_dict = {"accuracy_test" :[],
                    "accuracy_test_clean" :[],
                    "accuracy_test_response": [],
                    "f1_test" :[],
                    "f1_test_Clean":[],
                    "f1_response" :[]
}
for file_name_config in os.listdir(config_path):

    additional_fields = {
        "response": [],
        "adjacency_matrix_GT": [],
        "adjacency_matrix_test": [],
        "adjacency_matrix_test_clean": [],
        "adjacency_matrix_test": [],
        "adjacency_matrix_test_response":[],
        "accuracy_test": [],
        "accuracy_test_clean": [],
        "accuracy_response": [],
        "f1_test": [],
        "f1_test_clean": [],
        "f1_response": []
    }


    if file_name_config.endswith(".yaml"):
        file_config_path = os.path.join(config_path, file_name_config)
        with open(file_config_path, "r") as file:
                data = yaml.safe_load(file)
                goal = data['ACTIVITY_NAME']
                file_name = goal.replace(" ", "").lower()+'.json'


    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                # Load JSON data into the dictionary
                json_data = json.load(file)
                data[file_name] = json_data
        else:
            continue


    steps = data[file_name]['steps']
    #steps_str = "\n".join([f"{key}: {value}" for key, value in steps.items()])
    edges = data[file_name]['edges']
    num_steps = len(steps)
    adjacency_matrix_GT = np.zeros((num_steps, num_steps), dtype=int)
    adjacency_matrix_test = np.zeros((num_steps, num_steps), dtype=int)
    adjacency_matrix_test_response = np.zeros((num_steps, num_steps), dtype=int)





    steps_dic = {}
    for i_step,steps_key in enumerate(steps.keys()):
        steps_dic[steps_key] = i_step

    # Fill the adjacency matrix based on edges
    for edge in edges:
        from_step, to_step = edge
        adjacency_matrix_GT[from_step][to_step] = 1

    generate_graph(steps, adjacency_matrix_GT,output_folder_test,goal.replace(" ", "").lower()+'_GT')
    def convert_steps_to_indices(step_list,steps_dic):
        converted = [steps_dic[step_i] for step_i in step_list]
        # step_indices = {v.lower(): int(k) for k, v in steps.items()}
        # converted = []
        # for step in step_list:
        #     step_cleaned = step.strip().lower()
        #     if step_cleaned.isdigit():
        #         converted.append(int(step_cleaned))
        #     else:
        #         converted.append(step_indices.get(step_cleaned, -1))
        return converted



    steps_str = "\n".join([f"{key}: {value}" for key, value in data[file_name]['steps'].items()])
    data_output = data[file_name].copy()
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. Build a logical in sense of chronological sequence of the given bag of actions to get the goal. Use all of them. Given this sequence: ['0', '1', '5', '9', '12', '7', '11', '3', '6', '8', '10', '14', '2', '4', '15'], ['1', '5', '9', '12', '1', '7', '11', '3', '6', '8', '10', '14', '13', '4', '2', '15'],['0', '1', '5', '9', '12', '7', '11', '3', '6', '8', '10', '14', '4', '2', '15'] give me a different one. Return ONLY the numbers of the actions."},
            {"role": "user", "content": "goal: " + goal.format(str) + "bag of actions: (" + steps_str + ")"},

        ],
        temperature=0.1,
        max_tokens=500,
        stop=None,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
        top_logprobs=5,
    )
    response = completion.choices[0].message.content
    response_str = [item for item in re.split(r'[,\s]+', response) if item]
    print(response_str)
    #['0', '1', '5', '9', '12', '7', '11', '3', '6', '8', '10', '14', '2', '4', '15']
    #['0', '1', '5', '9', '12', '7', '11', '3', '6', '8', '10', '14', '2', '4', '15']
    #['1', '5', '9', '12', '1', '7', '11', '3', '6', '8', '10', '14', '13', '4', '2', '15']
    #['0', '1', '5', '9', '12', '7', '11', '3', '6', '8', '10', '14', '4', '2', '15']
    response_i = ''
    for i in data[file_name]['steps'].keys():
        steps_str = "\n".join([f"{key}: {value}" for key, value in data[file_name]['steps'].items() if not(key==i) and not(key=='0')])
        data_output = data[file_name].copy()
        value_step = data[file_name]['steps'][i]
        if value_step=='END':
            continue
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in partial ordering procedures. You will get a sequence of actions and a current action that is done. You need to choose possible next transition from given action it can be different than initial sequence you got. There might be a few possible answers. Choose from bag of actions, replay only the numbers of the actions . Take in to account the goal of the procedure. Return only the number of next possible action, no other words or signs"},
                {"role": "user", "content": "goal: " + goal.format(str)+ "sequence:" + str(response_str) +"current action:("+ value_step.format(str)+"), bag of actions: ("+steps_str+")"},

            ],
            temperature=0.1,
            max_tokens=500,
            stop=None,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=True,
            top_logprobs=5,
        )


        # Retrieve the response
        response = completion.choices[0].message.content
        response_str = [item for item in re.split(r'[,\s]+', response) if item]
        logprobs = completion.choices[0].logprobs



        # Print the resulting message
        response_i = response_i+value_step+"\n\n"
        print(value_step)
        print("\n\n")
        print(completion.choices[0].message.content)
        response_i = response_i + completion.choices[0].message.content + "\n\n"
        print("\n\n")

        token_vec = []
        probability_vec = []
        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        for token_i in top_logprobs:
            token_vec.append(token_i.token)
            probability = np.round(np.exp(token_i.logprob) * 100, 2)
            probability_vec.append(probability)
            response_i = response_i + f"  {token_i.token}: {token_i.logprob}" + f"  Probability: {probability}%"
            print(f"  {token_i.token}: {token_i.logprob}")
            print(f"  Probability: {probability}%")

        indices = np.where(np.array(probability_vec) > 0)[0]
        if len(indices)==1:
            probability_vec_keep = [token_vec[indices[0]]]
        else:
            probability_vec_keep = [token_vec[k] for k in indices]

        # check if the tokens exist in steps_dic
        for vec_keep_i in probability_vec_keep:
            if not(vec_keep_i in steps_dic.keys()):
                probability_vec_keep.pop(probability_vec_keep.index(vec_keep_i))


        converted_indices = convert_steps_to_indices(probability_vec_keep,steps_dic)
        current_action = convert_steps_to_indices([i],steps_dic)

        for next_possible_actions in converted_indices:
            adjacency_matrix_test[current_action[0],next_possible_actions] = 1

        for response_str_i in response_str:
            adjacency_matrix_test_response[current_action[0],int(response_str_i)] = 1

    additional_fields['response'] = response_i
    additional_fields['adjacency_matrix_GT'] = adjacency_matrix_GT.tolist()
    additional_fields['adjacency_matrix_test'] = adjacency_matrix_test.tolist()
    # Cleaning the adjacency_matrix_test
    adjacency_matrix_test_copy = np.copy(adjacency_matrix_test)
    adjacency_matrix_test_copy[:,0] = 0
    adjacency_matrix_test_copy[-1,:] = 0
    adjacency_matrix_test_response[:,0] = 0
    adjacency_matrix_test_response[-1, :] = 0
    adjacency_matrix_test[:, 0] = 0
    adjacency_matrix_test[-1, :] = 0

    for ii in range(num_steps):
        adjacency_matrix_test_copy[ii,ii] = 0
        adjacency_matrix_test_response[ii, ii] = 0
        adjacency_matrix_test[ii,ii] = 0

    sum_transitions = np.sum(adjacency_matrix_test_copy,0)
    sorted_indices = np.argsort(sum_transitions)
    sorted_indices_short = sorted_indices[1:-1]
    keep_transitions = []
    for column_i in sorted_indices_short:
        column_y = adjacency_matrix_test_copy[:, column_i]
        indices_of_ones = np.where(column_y == 1)[0]
        couples_permutations = list(permutations(indices_of_ones, 2))
        permutations_with_second = [(x, column_i) for x in indices_of_ones]
        keep_transitions.append(permutations_with_second)
        filtered_perm = [pair for pair in couples_permutations if pair not in keep_transitions]
        for perm_i in filtered_perm:
            adjacency_matrix_test_copy[perm_i] = 0


    for column_i in range(num_steps):
        column_y = adjacency_matrix_test_copy[:,column_i]
        compare_vec = (adjacency_matrix_test_copy[0,:]==column_y)
        adjacency_matrix_test_copy[0,compare_vec] = 0
        column_y==1

    generate_graph(steps, adjacency_matrix_test_copy, output_folder_test,goal.replace(" ", "").lower()+'_test_clean')
    generate_graph(steps, adjacency_matrix_test, output_folder_test,goal.replace(" ", "").lower()+'_test')
    generate_graph(steps, adjacency_matrix_test_response, output_folder_test,goal.replace(" ", "").lower()+'_test_response')
    additional_fields['adjacency_matrix_test_clean'] = adjacency_matrix_test_copy.tolist()
    additional_fields['adjacency_matrix_test_response'] = adjacency_matrix_test_response.tolist()
    additional_fields['accuracy_test'] = accuracy_score(adjacency_matrix_GT.flatten(),
                                                         adjacency_matrix_test.flatten())
    additional_fields['accuracy_test_clean'] = accuracy_score(adjacency_matrix_GT.flatten(), adjacency_matrix_test_copy.flatten())
    additional_fields['accuracy_response'] = accuracy_score(adjacency_matrix_GT.flatten(), adjacency_matrix_test_response.flatten())
    additional_fields['f1_test'] = f1_score(adjacency_matrix_GT.flatten(), adjacency_matrix_test.flatten())
    additional_fields['f1_test_clean'] = f1_score(adjacency_matrix_GT.flatten(), adjacency_matrix_test_copy.flatten())
    additional_fields['f1_response'] = f1_score(adjacency_matrix_GT.flatten(), adjacency_matrix_test_response.flatten())

    summary_dict["file_name"].append((file_name))
    summary_dict["accuracy_test"].append((additional_fields['accuracy_test']))
    summary_dict["accuracy_test_clean"].append((additional_fields['accuracy_test_clean']))
    summary_dict["accuracy_response"].append((additional_fields['accuracy_response']))
    summary_dict["f1_test"].append((additional_fields['f1_test']))
    summary_dict["f1_test_clean"].append((additional_fields['f1_test_clean']))
    summary_dict["f1_response"].append((additional_fields['f1_response']))

    data_output.update(additional_fields)
    # Create a new file name for the modified JSON
    new_file_name = file_name.replace(".json", "_test.json")
    new_file_path = os.path.join(output_folder_test, new_file_name)

    # Write the modified data to the new JSON file
    with open(new_file_path, "w") as new_file_name:
        json.dump(data_output, new_file_name, indent=4)

    test_vec.append(adjacency_matrix_test.flatten())
    test_clean_vec.append(adjacency_matrix_test_copy.flatten())
    test_response_vec.append(adjacency_matrix_test_response.flatten())
    GT_vec.append(adjacency_matrix_GT.flatten())

# Calculate accuracy
accuracy_test = accuracy_score(np.concatenate(GT_vec), np.concatenate(test_vec))
accuracy_test_clean = accuracy_score(np.concatenate(GT_vec), np.concatenate(test_clean_vec))
accuracy_test_response = accuracy_score(np.concatenate(GT_vec), np.concatenate(test_response_vec))
# Calculate F1-score
f1_test = f1_score(np.concatenate(GT_vec), np.concatenate(test_vec))
f1_test_clean = f1_score(np.concatenate(GT_vec), np.concatenate(test_clean_vec))
f1_response = f1_score(np.concatenate(GT_vec), np.concatenate(test_response_vec))

total_summary_dict['accuracy_test'] = accuracy_test
total_summary_dict['accuracy_test_clean'] = accuracy_test_clean
total_summary_dict['accuracy_test_response'] = accuracy_test_response
total_summary_dict['f1_test'] = f1_test
total_summary_dict['f1_test_Clean'] = f1_test_clean
total_summary_dict['f1_response'] = f1_response



new_file_name = "summary.json"
new_file_path = os.path.join(output_folder_test, new_file_name)
with open(new_file_path, "w") as new_file_name:
    json.dump(summary_dict, new_file_name, indent=4)

new_file_name = "summary_total.json"
new_file_path = os.path.join(output_folder_test, new_file_name)
with open(new_file_path, "w") as new_file_name:
    json.dump(total_summary_dict, new_file_name, indent=4)
y=1



#Loop through each token logprob and print all the top logprobs for each token
for token_logprob in completion.choices[0].logprobs.content:
    print(f"Token: {token_logprob.token}")
    print("Top Logprobs:")
    for top_logprob in token_logprob.top_logprobs:
        print(f"  {top_logprob.token}: {top_logprob.logprob}")
        probability = np.round(np.exp(top_logprob.logprob)*100,2)
        print(f"  Probability: {probability}%")
    print("\n")