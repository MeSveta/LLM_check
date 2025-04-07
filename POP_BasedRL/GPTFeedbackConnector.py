import openai
import os
import json
from openai import OpenAI


OPENAI_API_KEY = 'sk-proj-sbeE4x0JtQjYbNzvzU3ET3BlbkFJY2hPpO3T4X8WWUc61a47'

class GPTFeedbackConnector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_constraints_dot(self, actions, goal):
        """
        Given a bag of actions and a goal, use GPT to generate initial constraints and explanation.
        Return a dict: {"constraints": [...], "explanation": "..."}
        Includes validation and retries.
        """
        prompt = (
            f"Given the following goal: '{goal}', and the following actions:\n"
            f"{actions}\n"
            "Return the logical or temporal dependencies between these actions in a dot language as a list of [a, b] edges in the field \"constraints\",\n"
            "and provide a short explanation of the reasoning in the field \"explanation\".\n"
            "Return ONLY a JSON-compatible response like:\n"
            "{\"constraints\": [[0, 1], [1, 2], ...], \"explanation\": \"...\"}"
        )

        max_attempts = 3
        attempts = 0

        response = self._query_gpt(prompt)
        try:
            parsed = json.loads(response)
            cleaned_constraints = self.filter_constraints(parsed.get("constraints", []), actions)
            if self.validate_constraints(cleaned_constraints, actions):
                parsed["constraints"] = cleaned_constraints
                return parsed
        except Exception:
            pass


        return {
            "constraints": [],
            "explanation": "Invalid constraints format or action values after multiple attempts."
        }

    def evaluate_sequence(self, action_sequence, actions, goal):
        """
        Evaluate a final sequence of actions for chronological reasonableness.
        Return: {reward: 1 or 0, transitions: [problematic transitions] in dot format}
        """
        actions_text = [actions[str(i)] for i in action_sequence]
        """
         Evaluate a final sequence of actions for chronological reasonableness.
         Return: {reward: 1 or 0, transitions: [...], explanation: "..."}
         """
        actions_text = [str(i)+'-' + actions[str(i)] for i in action_sequence]
        #actions_text = [str(i) for i in action_sequence]
        prompt = (
            f"Given the following goal: '{goal}', and the following bag of actions:\n"
            f"{actions}\n"
            "Evaluate the following sequence of actions for achieving the goal, the sequence should include all bag actions:\n"
            f"{actions_text}\n"
            "The values at the beginning of the actions and sequence is dot language. Is the sequence chronologically reasonable?  Return the answer in JSON format as follows:\n"
            "{\n"
            "  \"reward\": 1 or 0,\n"
            "  \"good transitions\": [list only in order transition in the sequence at dot language, focus only on the right next transition whether it logical. if the sequence [0,4,7,8] and 4 can be right after 0 then [0,4] is in order transition. no "" before the numbers. The format [[0,4],]],\n"
            "  \"bad transitions\":  [list only out of order transition in the sequence at dot language, focus only on the right next transition whether it logical or not. if the sequence [0,4,7,8] and 4 cant be right after 0 then [0,4] is out of order transition. no "" before the numbers. The format [[0,4],]],\n"
            "  \"explanation\": \"Your reasoning here\"\n"
            "}"
        )

        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

            parsed = json.loads(raw_response)
            parsed_recheck = self.recheck_bad_transitions(goal, actions, action_sequence, parsed['bad transitions'])
            parsed['bad transitions'] = parsed_recheck['confirmed_bad_transitions']
            print("bad transitions")
            print(parsed['bad transitions'])
            return parsed

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

    def recheck_bad_transitions(self, goal, actions, sequence, bad_transitions):
        """
        Ask GPT to re-evaluate specific 'bad' transitions more carefully.
        """

        actions_text = [actions[str(i)] for i in sequence]
        prompt = (
            f"Given the goal: '{goal}' and the following actions:\n"
            f"{actions}\n\n"
            f"The sequence of actions (in dot IDs) was: {sequence}\n"
            f"The corresponding action names are: {actions_text}\n"
            f"The following transitions were marked as 'bad': {bad_transitions}\n\n"
            "Please recheck each bad transition one by one and evaluate if they are truly out of order.\n"
            "For each, respond with a JSON format only like this:\n"
            "{\n"
            "  \"confirmed_bad_transitions\": [[a, b], ...],\n"
            "  \"mistakenly_marked\": [[x, y], ...],\n"
            "  \"explanation\": \"Give a concise explanation per case\"\n"
            "}"
        )

        raw_response = self._query_gpt(prompt)

        try:
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

            parsed = json.loads(raw_response)
            return parsed

        except json.JSONDecodeError as e:
            print("❌ GPT output could not be parsed. Raw response:")
            print(raw_response)
            return {
                "confirmed_bad_transitions": [],
                "mistakenly_marked": [],
                "explanation": "Failed to parse GPT output"
            }



    def _query_gpt(self, prompt):
        """Internal helper to call the GPT API."""
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for understanding action plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error querying GPT: {str(e)}"

    def validate_constraints(self, parsed_constraints, steps):
        """Validate that all action indices used in constraints exist in the steps."""
        valid_keys = set(map(int, steps.keys()))
        return all(
            isinstance(pair, list) and
            len(pair) == 2 and
            all(isinstance(i, int) and i in valid_keys for i in pair)
            for pair in parsed_constraints
        )

    def filter_constraints(self, constraints, steps):
        """Remove constraint pairs that include any step not in the original list of steps."""
        valid_keys = set(map(int, steps.keys()))
        return [pair for pair in constraints if
                isinstance(pair, list) and len(pair) == 2 and all(isinstance(i, int) and i in valid_keys for i in pair)]


def main(input_folder, api_key=None):
    connector = GPTFeedbackConnector(api_key=api_key)

    output_folder = os.path.join(input_folder, "LLM")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

            steps = data.get("steps", {})
            goal = os.path.splitext(filename)[0]

            constraints = connector.generate_constraints_dot(steps, goal)
            data["constraints_LLM"] = constraints

            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as out_file:
                json.dump(data, out_file, indent=2)

    print(f"Processed files saved to: {output_folder}")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Run GPT constraint generation on JSON action files.")
    # parser.add_argument("folder", help="Path to the folder with JSON files")
    # parser.add_argument("--api_key", help="OpenAI API key (optional if set in environment)")
    # args = parser.parse_args()
    folder = r"C:\Users\Sveta\PycharmProjects\data\Cook"
    main(input_folder=folder)
