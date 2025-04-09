from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

model = SentenceTransformer('all-mpnet-base-v2')
query = "prepare a pancake"
query_emb = model.encode(query, convert_to_tensor=True)

#Suppose you have a list of concepts
concepts=["Add-1/2 tsp baking powder to a blender",
  "Serve-Serve the pancakes with chopped strawberries",
   "Melt-Melt a small knob of butter in a non-stick frying pan over low-medium heat",
     "splash-splash maple syrup on plate",
    "Add-Add 1 banana to a blender",
    "Cook-Cook for 1 min or until the tops start to bubble",
    "blitz-blitz the blender for 20 seconds",
    "Flip-Flip the pancakes with a fork or a fish slice spatula",
   "Add-1 egg to a blender",
    "cook-cook for 20-30 seconds more",
    "Pour-Pour three little puddles straight from the blender into the frying pan",
    "Add-1 heaped tbsp flour to a blender",
    "Chop-Chop 1 strawberry",
    "Transfer-Transfer to a plate"]
    #       , "Add-1/2 tsp baking powder to a blender, Add-Add 1 banana to a blender,Add-1 egg to a blender,Add-1 heaped tbsp flour to a blender", "Serve-Serve the pancakes with chopped strawberries, Add-1/2 tsp baking powder to a blender, Add-Add 1 banana to a blender,Add-1 egg to a blender,Add-1 heaped tbsp flour to a blender","Add-1/2 tsp baking powder to a blender, Add-Add 1 banana to a blender,Add-1 egg to a blender,Add-1 heaped tbsp flour to a blender, Serve-Serve the pancakes with chopped strawberries","car engine, Add-1/2 tsp baking powder to a blender, Add-Add 1 banana to a blender,Add-1 egg to a blender,Add-1 heaped tbsp flour to a blender",)
    # "Add-1/2 tsp baking powder to a blender, Add-Add 1 banana to a blender,Add-1 egg to a blender,Add-1 heaped tbsp flour to a blender, blitz-blitz the blender for 20 seconds, Serve-Serve the pancakes with chopped strawberries"     ]

#concepts = ["flip the pancake", "heat the pan", "car engine", "tire pressure", "whisk the eggs", "flip the pancake, whisk the eggs", "whisk the eggs, flip the pancake", "tire pressure, whisk the eggs, flip the pancake",  "whisk the eggs, tire pressure, flip the pancake"]
concept_embs = model.encode(concepts, convert_to_tensor=True)

# Cosine similarity
cos_scores = util.cos_sim(query_emb, concept_embs)
top_results = torch.topk(cos_scores, k=3)

# Get top-k relevant concepts
for score, idx in zip(top_results[0][0], top_results[1][0]):
    print(f"{concepts[idx]}: {score:.4f}")



length = np.linalg.norm(query_emb)
print(f"Length of embedding for '{query}':", length)


# Step 4: Define MI estimator
def estimate_mi_by_dimension(v1, v2, bins=20):
    v1_disc = np.digitize(v1, np.histogram(v1, bins=bins)[1][:-1])
    v2_disc = np.digitize(v2, np.histogram(v2, bins=bins)[1][:-1])
    return mutual_info_score(v1_disc, v2_disc)

# Step 5: Calculate MI scores
mi_scores = [estimate_mi_by_dimension(query_emb, c_emb) for c_emb in concept_embs]

# Step 6: Show results
for concept, mi in zip(concepts, mi_scores):
    print(f"{concept}: MI = {mi:.4f}")


def normalize(v):
    return v / np.linalg.norm(v)

goal_dir = normalize(query_emb)
concept_dirs = [normalize(v) for v in concept_embs]

# Stack for PCA
all_dirs = np.vstack([goal_dir] + concept_dirs)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_dirs)

# Plot
plt.figure(figsize=(8, 8))
origin = np.array([[0, 0]] * len(reduced))
labels = ["GOAL: prepare a pancake"] + concepts

# Arrows from origin to direction vector
for i, label in enumerate(labels):
    plt.arrow(0, 0, reduced[i, 0], reduced[i, 1],
              head_width=0.02, length_includes_head=True, color='tab:blue' if i == 0 else 'tab:green')
    plt.text(reduced[i, 0]*1.05, reduced[i, 1]*1.05, label, fontsize=9)

plt.title("Semantic Directions of Concepts (PCA-reduced)")
plt.grid(True)
plt.axis('equal')
plt.show()


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentence = "She sat by the bank"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# Outputs: last hidden states (sequence of contextual embeddings)
embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)

y=1
