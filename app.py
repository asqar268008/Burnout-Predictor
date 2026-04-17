import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from Burnout import BurnoutPredictorInput
import joblib

app = FastAPI()

model = joblib.load('burnout.pkl')

classifier = model['model']
encoder = model['encoder']

@app.post('/predict')
def predict_burnout(data: BurnoutPredictorInput):
    data = data.dict()
    age = data['age']
    experience_years = data['experience_years']
    daily_work_hours = data['daily_work_hours']
    sleep_hours = data['sleep_hours']
    caffeine_intake = data['caffeine_intake']
    bugs_per_day = data['bugs_per_day']
    commits_per_day = data['commits_per_day']
    meetings_per_day = data['meetings_per_day']
    screen_time = data['screen_time']
    exercise_hours = data['exercise_hours']
    stress_level = data['stress_level']
    prediction = classifier.predict([[age, experience_years, daily_work_hours, sleep_hours, caffeine_intake,
                            bugs_per_day, commits_per_day, meetings_per_day, screen_time,
                            exercise_hours, stress_level]])
    result = encoder.inverse_transform(prediction)
    return {
        "prediction_numeric": int(prediction[0]),
        "prediction_label": result[0]
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5001)