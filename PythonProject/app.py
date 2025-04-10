from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and preprocessor
dtr = pickle.load(open("dtr.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Create Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Predict", methods=["POST"])
def Predict():
    if request.method == "POST":
        # Get form values
        Crop = request.form["Crop"].strip()
        Crop_Year = float(request.form["Crop_Year"])
        Season = request.form["Season"].strip()
        State = request.form["State"].strip()
        Area = float(request.form["Area"])
        Annual_Rainfall = float(request.form["Annual_Rainfall"])
        Fertilizer = float(request.form["Fertilizer"])
        Pesticide = float(request.form["Pesticide"])

        # Load encoders and scaler from preprocessor
        scaler = preprocessor["scaler"]
        label_encoders = preprocessor["label_encoders"]

        # Apply label encoding to categorical features
        try:
            Crop = label_encoders["Crop"].transform([Crop])[0]
            Season = label_encoders["Season"].transform([Season])[0]
            State = label_encoders["State"].transform([State])[0]
        except ValueError as e:
            return render_template("index.html", prediction=f"Invalid input: {e}")

        # Build feature vector (without Production)
        features = np.array([[Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict both Yield and Production
        prediction = dtr.predict(features_scaled)[0]  # Expected output: [Yield, Production]

        return render_template("index.html",
                               yield_prediction=round(prediction[0], 2),
                               production_prediction=round(prediction[1], 2))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
