import pandas as pd
import re
import joblib


# Function to parse Ludii data
def parse_ludii_data(file_path):
    """
    Reads and parses the input Ludii data file.

    Parameters:
        file_path (str): Path to the input.txt file containing the game description.

    Returns:
        dict: Parsed components of the game description.
    """
    with open(file_path, 'r') as file:
        ludii_data = file.read()

    patterns = {
        "game": r'\(game\s+"(.*?)"',
        "players": r'\(players\s+(\d+)\)',
        "equipment": r'\(equipment\s+{(.*?)}\s*\)',
        "rules": r'\(rules\s+\((.*)\)\s*\)'
    }
    parsed_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, ludii_data, re.DOTALL)
        parsed_data[key] = match.group(1).strip() if match else None
    return parsed_data


# Function to extract features from parsed Ludii data
def extract_ludii_features(parsed_data):
    """
    Extracts features from parsed Ludii data.

    Parameters:
        parsed_data (dict): Parsed Ludii data.

    Returns:
        dict: Features extracted from the Ludii game description.
    """
    features = {}
    features['game_name'] = parsed_data['game']
    features['num_players'] = pd.to_numeric(parsed_data['players'], errors='coerce')
    features['num_pieces'] = len(re.findall(r'\(piece', parsed_data['equipment'])) if parsed_data['equipment'] else 0
    features['board_type'] = re.search(r'\(board \((.*?)\)', parsed_data['equipment']).group(1) if parsed_data['equipment'] and re.search(r'\(board \((.*?)\)', parsed_data['equipment']) else 'Unknown'
    features['num_conditions'] = parsed_data['rules'].count('if') if parsed_data['rules'] else 0
    features['num_moves'] = parsed_data['rules'].count('move') if parsed_data['rules'] else 0
    features['num_triggers'] = parsed_data['rules'].count('trigger') if parsed_data['rules'] else 0
    features['rule_complexity'] = len(re.findall(r'[<>=%]', parsed_data['rules'])) if parsed_data['rules'] else 0
    return features


# Function to parse agent string
def parse_agent(agent_string):
    """
    Parses the agent string into its components.

    Parameters:
        agent_string (str): The agent string description.

    Returns:
        dict: Parsed components of the agent string.
    """
    components = agent_string.split('-')
    if len(components) != 5:
        return {
            'selection': None,
            'exploration_const': None,
            'playout': None,
            'score_bounds': None
        }
    return {
        'selection': components[1],
        'exploration_const': float(components[2]),
        'playout': components[3],
        'score_bounds': components[4] == 'true'  # Convert to boolean
    }


# Function to run inference
def run_inference(input_file, agent_string, model_path):
    """
    Runs inference on the given input data and agent string.

    Parameters:
        input_file (str): Path to the input.txt file containing the game description.
        agent_string (str): String description of the agent.
        model_path (str): Path to the pre-trained model.

    Returns:
        float: Predicted utility value.
    """
    # Parse Ludii data and agent string
    parsed_ludii_data = parse_ludii_data(input_file)
    ludii_features = extract_ludii_features(parsed_ludii_data)
    agent_features = parse_agent(agent_string)

    # Combine all features into a single dictionary
    features = {**ludii_features, **agent_features}

    # Convert to DataFrame for model compatibility
    inference_data = pd.DataFrame([features])

    # Load the pre-trained model
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Ensure columns align with training data
    required_columns = [
        'game_name', 'num_players', 'num_pieces', 'board_type',
        'num_conditions', 'num_moves', 'num_triggers', 'rule_complexity',
        'selection', 'exploration_const', 'playout', 'score_bounds'
    ]
    for col in required_columns:
        if col not in inference_data.columns:
            inference_data[col] = 0  # Add missing columns with default values

    # Run prediction
    prediction = loaded_model.predict(inference_data)
    return prediction[0]


# Example usage

agent_strings = ['MCTS-UCB1-0.1-Random200-true', 'MCTS-UCB1-0.1-Random200-false', 'MCTS-UCB1-0.1-MAST-true', 'MCTS-UCB1-0.1-MAST-false', 'MCTS-UCB1-0.1-NST-true', 'MCTS-UCB1-0.1-NST-false', 'MCTS-UCB1-0.6-Random200-true', 'MCTS-UCB1-0.6-Random200-false', 'MCTS-UCB1-0.6-MAST-true', 'MCTS-UCB1-0.6-MAST-false', 'MCTS-UCB1-0.6-NST-true', 'MCTS-UCB1-0.6-NST-false', 'MCTS-UCB1-1.41421356237-Random200-true', 'MCTS-UCB1-1.41421356237-Random200-false', 'MCTS-UCB1-1.41421356237-MAST-true', 'MCTS-UCB1-1.41421356237-MAST-false', 'MCTS-UCB1-1.41421356237-NST-true', 'MCTS-UCB1-1.41421356237-NST-false', 'MCTS-UCB1GRAVE-0.1-Random200-true', 'MCTS-UCB1GRAVE-0.1-Random200-false', 'MCTS-UCB1GRAVE-0.1-MAST-true', 'MCTS-UCB1GRAVE-0.1-MAST-false', 'MCTS-UCB1GRAVE-0.1-NST-true', 'MCTS-UCB1GRAVE-0.1-NST-false', 'MCTS-UCB1GRAVE-0.6-Random200-true', 'MCTS-UCB1GRAVE-0.6-Random200-false', 'MCTS-UCB1GRAVE-0.6-MAST-true', 'MCTS-UCB1GRAVE-0.6-MAST-false', 'MCTS-UCB1GRAVE-0.6-NST-true', 'MCTS-UCB1GRAVE-0.6-NST-false', 'MCTS-UCB1GRAVE-1.41421356237-Random200-true', 'MCTS-UCB1GRAVE-1.41421356237-Random200-false', 'MCTS-UCB1GRAVE-1.41421356237-MAST-true', 'MCTS-UCB1GRAVE-1.41421356237-MAST-false', 'MCTS-UCB1GRAVE-1.41421356237-NST-true', 'MCTS-UCB1GRAVE-1.41421356237-NST-false', 'MCTS-ProgressiveHistory-0.1-Random200-true', 'MCTS-ProgressiveHistory-0.1-Random200-false', 'MCTS-ProgressiveHistory-0.1-MAST-true', 'MCTS-ProgressiveHistory-0.1-MAST-false', 'MCTS-ProgressiveHistory-0.1-NST-true', 'MCTS-ProgressiveHistory-0.1-NST-false', 'MCTS-ProgressiveHistory-0.6-Random200-true', 'MCTS-ProgressiveHistory-0.6-Random200-false', 'MCTS-ProgressiveHistory-0.6-MAST-true', 'MCTS-ProgressiveHistory-0.6-MAST-false', 'MCTS-ProgressiveHistory-0.6-NST-true', 'MCTS-ProgressiveHistory-0.6-NST-false', 'MCTS-ProgressiveHistory-1.41421356237-Random200-true', 'MCTS-ProgressiveHistory-1.41421356237-Random200-false', 'MCTS-ProgressiveHistory-1.41421356237-MAST-true', 'MCTS-ProgressiveHistory-1.41421356237-MAST-false', 'MCTS-ProgressiveHistory-1.41421356237-NST-true', 'MCTS-ProgressiveHistory-1.41421356237-NST-false', 'MCTS-UCB1Tuned-0.1-Random200-true', 'MCTS-UCB1Tuned-0.1-Random200-false', 'MCTS-UCB1Tuned-0.1-MAST-true', 'MCTS-UCB1Tuned-0.1-MAST-false', 'MCTS-UCB1Tuned-0.1-NST-true', 'MCTS-UCB1Tuned-0.1-NST-false', 'MCTS-UCB1Tuned-0.6-Random200-true', 'MCTS-UCB1Tuned-0.6-Random200-false', 'MCTS-UCB1Tuned-0.6-MAST-true', 'MCTS-UCB1Tuned-0.6-MAST-false', 'MCTS-UCB1Tuned-0.6-NST-true', 'MCTS-UCB1Tuned-0.6-NST-false', 'MCTS-UCB1Tuned-1.41421356237-Random200-true', 'MCTS-UCB1Tuned-1.41421356237-Random200-false', 'MCTS-UCB1Tuned-1.41421356237-MAST-true', 'MCTS-UCB1Tuned-1.41421356237-MAST-false', 'MCTS-UCB1Tuned-1.41421356237-NST-true', 'MCTS-UCB1Tuned-1.41421356237-NST-false', 'MCTS-ProgressiveWidening-0.1-Random200-true', 'MCTS-ProgressiveWidening-0.1-Random200-false', 'MCTS-ProgressiveWidening-0.1-MAST-true', 'MCTS-ProgressiveWidening-0.1-MAST-false', 'MCTS-ProgressiveWidening-0.1-NST-true', 'MCTS-ProgressiveWidening-0.1-NST-false', 'MCTS-ProgressiveWidening-0.6-Random200-true', 'MCTS-ProgressiveWidening-0.6-Random200-false', 'MCTS-ProgressiveWidening-0.6-MAST-true', 'MCTS-ProgressiveWidening-0.6-MAST-false', 'MCTS-ProgressiveWidening-0.6-NST-true', 'MCTS-ProgressiveWidening-0.6-NST-false', 'MCTS-ProgressiveWidening-1.41421356237-Random200-true', 'MCTS-ProgressiveWidening-1.41421356237-Random200-false', 'MCTS-ProgressiveWidening-1.41421356237-MAST-true', 'MCTS-ProgressiveWidening-1.41421356237-MAST-false', 'MCTS-ProgressiveWidening-1.41421356237-NST-true', 'MCTS-ProgressiveWidening-1.41421356237-NST-false']
if __name__ == "__main__":
    input_file_path = "./input/input.txt"  # Path to the input.txt file
    agent_string = "MCTS-ProgressiveHistory-0.1-MAST-true"  # Example agent string
    model_path = "./models/lightgbm_model.pkl"  # Path to the saved model

    best_agent_string = ""
    best_utility = -float('inf')
    for agent_string in agent_strings:
        # Run inference
        curr_utility = run_inference(input_file_path, best_agent_string, model_path)
        if curr_utility > best_utility:
            best_utility = curr_utility
            best_agent_string = agent_string

    print(best_agent_string)
    predicted_utility = run_inference(input_file_path, best_agent_string, model_path)
    print(f"Predicted Utility: {predicted_utility}")