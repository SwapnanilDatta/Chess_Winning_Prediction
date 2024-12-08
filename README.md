# Chess Winning Prediction Project

## Overview
This project predicts the outcome of chess games and the expected number of turns based on player ratings and the opening strategy. Using a dataset of 10,000 games from **Lichess**, we applied feature engineering to refine our data to 3,600+ entries. 

To address model overfitting during the initial experiments, we employed **GridSearchCV** to tune hyperparameters and improve performance. The project utilizes separate models for:
1. Predicting the number of turns (regression task).
2. Predicting the winner of the game (classification task).

---

## Problem Statement
The goal is to:
- Predict the **number of turns** in a chess game based on player ratings and opening name.
- Predict the **winner** (White, Black, or Draw) with probabilities for each outcome.

---

## Dataset
- **Source:** [Lichess Games Dataset](https://database.lichess.org/)
- **Initial Size:** 10,000 games
- **Processed Size:** 3,600+ entries after grouping and feature engineering.

### Features Used
- **`white_rating`**: Elo rating of the white player.
- **`black_rating`**: Elo rating of the black player.
- **`opening_name`**: Name of the opening strategy (encoded for modeling).

### Targets
- **`turns`**: Number of turns in the game (regression target).
- **`winner`**: Winner of the game (White, Black, or Draw) (classification target).

---

## Models
### 1. Turns Prediction (Regression)
- **Model**: RandomForestRegressor
- **Hyperparameter Tuning**: GridSearchCV
  - `n_estimators`: [50, 100, 150]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5, 10]

### 2. Winner Prediction (Classification)
- **Model**: RandomForestClassifier
- **Hyperparameter Tuning**: GridSearchCV
  - `n_estimators`: [50, 100, 150]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5, 10]

### Best Classification Parameters:
```json
{
  "model__max_depth": 10,
  "model__min_samples_split": 10,
  "model__n_estimators": 150
}
```

---

## Evaluation Metrics
### Classification Report:
| Label      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Black** (0) | 0.61      | 0.55   | 0.58     | 332     |
| **White** (1) | 0.60      | 0.73   | 0.66     | 353     |
| **Draw** (2)  | 0.00      | 0.00   | 0.00     | 40      |

**Accuracy**: 60%  
**Weighted Avg F1-Score**: 58%

---

## Project Pipeline
1. **Data Preprocessing**:
   - Encoded categorical features (`opening_name` and `winner`).
   - Split data into training and testing sets.

2. **Modeling**:
   - Trained separate regression and classification models using GridSearchCV for optimal hyperparameter tuning.

3. **Deployment**:
   - Created a Flask web app to serve predictions.
   - Accepts `white_rating`, `black_rating`, and `opening_name` as inputs.
   - Outputs:
     - Predicted **number of turns**.
     - Probabilities for **winner** (White, Black, Draw).

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone <repository-url>
cd chess-winning-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask App
```bash
python app.py
```

### 4. Access the Web App
- Open a browser and go to `http://127.0.0.1:5000/`
- Use the following JSON format to make predictions:
```json
{
  "white_rating": 1500,
  "black_rating": 1450,
  "opening_name": "Sicilian Defense"
}
```

---

## Future Improvements
- Handle imbalanced classes for better draw prediction.
- Include additional features like time control, player country, and match type.
- Explore advanced models such as Gradient Boosting or Neural Networks.

---

## Acknowledgments
- **Lichess** for the dataset.
- **Scikit-Learn** for modeling and evaluation tools.
- **Flask** for deployment.

---

Feel free to reach out for suggestions or feedback!
