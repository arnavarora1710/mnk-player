import pandas as pd
import re
import joblib
import enum as Enum

class SelectionStrategy(Enum):
    UCB1 = "UCB1"
    UCB1GRAVE = "UCB1GRAVE"
    ProgressiveHistory = "ProgressiveHistory"
    UCB1Tuned = "UCB1Tuned"
    ProgressiveWidening = "ProgressiveWidening"

class ExplorationConst(Enum):
    CONST_0_1 = 0.1
    CONST_0_6 = 0.6
    CONST_1_41 = 1.41421356237

class PlayoutStrategy(Enum):
    Random200 = "Random200"
    MAST = "MAST"
    NST = "NST"

class ScoreBounds(Enum):
    TRUE = "true"
    FALSE = "false"

def generate_all_strings():
    all_strings = []
    for selection in SelectionStrategy:
        for exploration in ExplorationConst:
            for playout in PlayoutStrategy:
                for score in ScoreBounds:
                    all_strings.append(f"MCTS-{selection.value}-{exploration.value}-{playout.value}-{score.value}")
    return all_strings

def read_ludii_data(file_path):
    with open(file_path, 'r') as file:
        return file.read()

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

input_file_path = "./input/input.txt"
ludii_data = read_ludii_data(input_file_path)

parsed_data = parse_ludrules(ludii_data)
features = extract_features(parsed_data)

inference_data = pd.DataFrame([features])

model_path = "./models/gradient_boosting_model.pkl"
loaded_model = joblib.load(model_path)

required_columns = [
    'game_name', 'num_players', 'num_pieces', 'board_type', 
    'num_conditions', 'num_moves', 'num_triggers', 'rule_complexity'
]
for col in required_columns:
    if col not in inference_data.columns:
        inference_data[col] = 0

predictions = loaded_model.predict(inference_data)
print(f"Predicted Utility: {predictions[0]}")