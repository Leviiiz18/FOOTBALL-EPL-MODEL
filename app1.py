from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

matches = pd.read_csv('epldata.csv')
matches.dropna(inplace=True)

# Feature engineering
matches['goal_difference'] = matches['FTHG'] - matches['FTAG']
matches['home_team_form'] = matches.groupby('HomeTeam')['goal_difference'].rolling(5).mean().reset_index(level=0, drop=True)
matches['away_team_form'] = matches.groupby('AwayTeam')['goal_difference'].rolling(5).mean().reset_index(level=0, drop=True)

features = ['HomeTeam', 'AwayTeam', 'home_team_form', 'away_team_form']
X = matches[features]
X.fillna(0, inplace=True)

# Preprocessing pipeline
numeric_features = ['home_team_form', 'away_team_form']
categorical_features = ['HomeTeam', 'AwayTeam']
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)])

# Split the data for training
y_home = matches['FTHG']
y_away = matches['FTAG']

X_train = preprocessor.fit_transform(X)

# Train the models
home_goal_model = RandomForestRegressor(n_estimators=100, random_state=42)
home_goal_model.fit(X_train, y_home)

away_goal_model = RandomForestRegressor(n_estimators=100, random_state=42)
away_goal_model.fit(X_train, y_away)

# Function to predict the match outcome
def predict_match(HomeTeam, AwayTeam):
    # Calculate the form of both teams
    home_team_form = matches[matches['HomeTeam'] == HomeTeam]['goal_difference'].mean()
    away_team_form = matches[matches['AwayTeam'] == AwayTeam]['goal_difference'].mean()

    # Create a new match data
    new_match = pd.DataFrame({
        'HomeTeam': [HomeTeam],
        'AwayTeam': [AwayTeam],
        'home_team_form': [home_team_form],
        'away_team_form': [away_team_form]
    })

    new_match_preprocessed = preprocessor.transform(new_match)

    # Predict goals
    home_goals = int(home_goal_model.predict(new_match_preprocessed)[0])
    away_goals = int(away_goal_model.predict(new_match_preprocessed)[0])

    # Determine outcome
    if home_goals > away_goals:
        result = f"{HomeTeam} wins"
    elif home_goals < away_goals:
        result = f"{AwayTeam} wins"
    else:
        result = "Draw"

    return result, home_goals, away_goals

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['homeTeam']
    away_team = request.form['awayTeam']

    result, home_goals, away_goals = predict_match(home_team, away_team)

    return jsonify({'result': result, 'home_goals': home_goals, 'away_goals': away_goals})

if __name__ == '__main__':
    app.run(debug=True)
