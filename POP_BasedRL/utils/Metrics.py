from POP_BasedRL.RLAgent import RLAgent

class Metrics:
    def __init__(self,q_table, true_constrains):
        self.q_table = q_table
    # === Metrics for Evaluating Learned Constraints ===
    def extract_learned_policy(q_table):
        """Extract greedy policy from Q-table."""
        policy = {}
        for state, actions in q_table.items():
            if actions:
                best_action = max(actions, key=actions.get)
                policy[state] = best_action
        return policy

    def compare_to_true_constraints(learned_policy, true_edges):
        """Compute precision, recall, F1 against true constraints."""
        learned_edges = set((s, a) for s, a in learned_policy.items())
        true_edges = set(true_edges)

        tp = len(learned_edges & true_edges)
        fp = len(learned_edges - true_edges)
        fn = len(true_edges - learned_edges)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "learned_edges": learned_edges
        }