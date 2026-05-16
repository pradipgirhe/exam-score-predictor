# Student Exam Score Prediction System

A Machine Learning web application built using Flask that predicts a student's exam score based on study-related inputs.

The system uses a trained Machine Learning model and scaler to predict exam performance.

---

# Features

- Predicts student exam score
- Simple Flask web interface
- Uses Machine Learning model
- Input scaling using StandardScaler
- Real-time prediction
- Beginner friendly project

---

# Technologies Used

- Python
- Flask
- NumPy
- Scikit-learn
- Pickle
- HTML/CSS

---

# Machine Learning Inputs

The prediction is based on:

- Hours Studied
- Hours of Sleep
- Attendance Percentage
- Previous Exam Score

---

# Project Structure

```text
student-score-predictor/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── templates/
│   └── index.html
├── static/
│
├── README.md
```

---

# Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/student-score-predictor.git
```

---

## Step 2: Open Project Folder

```bash
cd student-score-predictor
```

---

## Step 3: Install Required Libraries

```bash
pip install flask numpy scikit-learn
```

---

# How to Run

Run the Flask application:

```bash
python app.py
```

OR

```bash
python3 app.py
```

---

# Open in Browser

After running the project, open:

```text
http://127.0.0.1:5000/
```

---

# How It Works

1. User enters:
   - Study hours
   - Sleep hours
   - Attendance
   - Previous score

2. Data is converted into numerical array

3. Input data is scaled using saved scaler

4. Machine Learning model predicts exam score

5. Predicted result is displayed on webpage

---

# Example Input

| Feature | Value |
|---|---|
| Hours Studied | 5 |
| Hours Sleep | 7 |
| Attendance | 85 |
| Previous Score | 78 |

---

# Example Output

```text
Predicted Exam Score: 82.45
```

---

# Modules Used

```python
from flask import Flask, render_template, request
import numpy as np
import pickle
```

---

# Future Improvements

- Add student login system
- Store prediction history
- Deploy project online
- Add charts and analytics
- Improve UI design
- Use advanced ML algorithms

---

# Deployment Platforms

You can deploy this project for free on:

- Render
- Railway
- Vercel
- PythonAnywhere

---

# Author

Pradip Girhe

GitHub:
https://github.com/pradipgirhe

---

# License

This project is open-source and free to use.
