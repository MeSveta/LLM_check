import json
import os
from collections import defaultdict, deque

def generate_and_save_valid_sequence(filepath):
    # Load JSON data
    with open(filepath, 'r') as file:
        data = json.load(file)

    edges = data["edges"]
    steps = data["steps"]

    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)

    # Include all nodes from steps even if they are disconnected
    for node in steps.keys():
        nodes.add(int(node))

    queue = deque()
    for node in sorted(nodes):  # Sorting ensures deterministic output
        if in_degree[node] == 0:
            queue.append(node)

    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(steps):
        raise ValueError(f"Graph in {filepath} contains a cycle or disconnected components")

    # Save the result back into the file
    data["valid_sequence"] = result

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2)

    print(f"✅ Valid sequence added to: {filepath}")


def main(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                generate_and_save_valid_sequence(filepath)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")


# Example usage
if __name__ == "__main__":

    folder =  "C:/Users/Sveta/PycharmProjects/data/Cook/LLM"
    main(folder)
