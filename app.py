from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

DATASETS = {
    "2023": {
        "ncaa": "templates/datasets/2023NCAA_Combined_Stats.csv",
        "rmac": "templates/datasets/rmac_2023_dataset.csv"
    },
    "2024": {
        "ncaa": "templates/datasets/2024NCAA_Combined_Stats.csv",
        "rmac": "templates/datasets/rmac_2024_dataset.csv"
    },
    "2025": {
        "ncaa": "templates/datasets/2025NCAA_Combined_Stats.csv",
        "rmac": "templates/datasets/rmac_2025_dataset.csv"
    }
}

@app.route("/")
def home():
    return render_template("FinalProject.html")


# -----------------------------------------------------------
# VIEW DATA TAB
# -----------------------------------------------------------
@app.route("/view_data", methods=["POST"])
def view_data():
    year = request.form.get("year")
    conference = request.form.get("conference")

    data_path = DATASETS[year][conference]
    df = pd.read_csv(data_path)

    return df.to_html()


# Merge NCAA and RMAC datasets together
def load_and_merge_year(year):
    ncaa = pd.read_csv(DATASETS[year]["ncaa"])
    rmac = pd.read_csv(DATASETS[year]["rmac"])

    # Clean team names in NCAA
    ncaa['Team'] = ncaa['Team'].str.replace(r"\([^)]*\)", "", regex=True).str.strip()

    ncaa['Team'] = ncaa['Team'].replace({
        'UC-Colo. Springs': 'UCCS',
        'Colo. Sch. of Mines': 'Colorado School of Mines',
        'Black Hills St.': 'Black Hills State',
        'N.M. Highlands': 'New Mexico Highlands',
        'Adams St.': 'Adams State',
        'Chadron St.': 'Chadron State'
    })

    for col in ['PO', 'A', 'E']:
        ncaa[col] = pd.to_numeric(ncaa[col], errors='coerce')


    # Fielding percentage
    ncaa['FP'] = (ncaa['PO'] + ncaa['A']) / (ncaa['PO'] + ncaa['A'] + ncaa['E'])

    merged = pd.merge(rmac, ncaa, on="Team", how="left")
    merged = merged.fillna(0)

    return merged


# Learning model
@app.route("/Make_a_Prediction", methods=["POST"])
def make_prediction():
    data = request.get_json()
    years = data.get("years", [])

    if len(years) <= 1:
        return jsonify({"error": "You must select 2 or more years to predict a new season."}), 400


    # Here, I'm training the model on all years, 2023-2025
    train_frames = []
    for y in years:
        train_frames.append(load_and_merge_year(y))

    train_df = pd.concat(train_frames, ignore_index=True)

    # Features used for training
    features = ['W', 'L', 'PCT_x', 'BA', 'FP', 'ERA']
    features = [f for f in features if f in train_df.columns]  # remove missing features

    X_train = train_df[features].fillna(0)
    y_train = train_df['Rank'].fillna(0)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
#-----------------------------------------------------------------------------
    # Prediction on the most recent dataset
    latest_year = sorted(years)[-1]

    predict_df = load_and_merge_year(latest_year)
    predict_X = predict_df[features].fillna(0)

    # Predict ONCE per team (exactly 12 teams)
    y_pred = model.predict(predict_X)

    # Extract teams for final output
    teams = predict_df["Team"].tolist()

    train_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, train_pred)
    r2 = r2_score(y_train, train_pred)

    # Print results
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    # output being sent back to javascript
    return jsonify({
        "teams": teams,
        "predictions": y_pred.tolist(),
        "r2": r2,
        "mae": mae,
        "year_predicted": latest_year
    })

if __name__ == "__main__":
    app.run(debug=False)

