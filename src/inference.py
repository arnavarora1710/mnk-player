import pandas as pd
import re
import joblib

# Function to read Ludii data from input.txt
def read_ludii_data(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Parsing Ludii rules
def parse_ludrules(ludrules):
    parsed_data = {}
    patterns = {
        "game": r'\(game\s+"(.*?)"',
        "players": r'\(players\s+(\d+)\)',
        "equipment": r'\(equipment\s+{(.*?)}\s*\)',
        "rules": r'\(rules\s+\((.*)\)\s*\)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, ludrules, re.DOTALL)
        parsed_data[key] = match.group(1).strip() if match else None
    return parsed_data

# Extract features from parsed Ludii rules
def extract_features(parsed_data):
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

# Load Ludii data from input.txt
input_file_path = "./input/input.txt"  # Path to the input.txt file
ludii_data = read_ludii_data(input_file_path)

# Parse and extract features
parsed_data = parse_ludrules(ludii_data)
features = extract_features(parsed_data)

# Convert features to DataFrame for model compatibility
inference_data = pd.DataFrame([features])

# Load the trained model
model_path = "./models/gradient_boosting_model.pkl"  # Replace with your model's path
loaded_model = joblib.load(model_path)

# Ensure the columns align with training data preprocessing
required_columns = [
    'game_name', 'num_players', 'num_pieces', 'board_type', 
    'num_conditions', 'num_moves', 'num_triggers', 'rule_complexity'
]
for col in required_columns:
    if col not in inference_data.columns:
        inference_data[col] = 0  # Add missing columns with default values

# Run inference
predictions = loaded_model.predict(inference_data)
print(f"Predicted Utility: {predictions[0]}")