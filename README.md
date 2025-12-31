## Iris Classification using Machine Learning and Flask

This project is an end-to-end Machine Learning application that classifies Iris flower species using Logistic Regression and deploys the trained model through a Flask web application for real-time predictions.

---

# Project Overview

The Iris Classification system predicts the species of an Iris flower — **Setosa, Versicolor, or Virginica** — based on four input features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The trained model is integrated into a Flask web app where users can enter measurements and instantly get predictions.

---

# Machine Learning Model

- Algorithm: **Logistic Regression**
- Dataset: **Iris Dataset**
- Train-Test Split: **80% / 20%**
- Feature Scaling: **StandardScaler**
- Accuracy Achieved: **100%**

---

# Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Flask
- Joblib
- HTML, CSS
- Google Colab
- GitHub

---

# Project Structure
```
Iris_classification-system
├── app.py
├── model.pkl
├── scaler.pkl
├── templates/
│ └── index.html
├── static/
│ └── style.css
├── Iris_Classification.ipynb
├── requirements.txt
└── README.md
```

---

# How to Run the Project

1. Clone the repository
```
git clone https://github.com/your-username/Iris-classification-flask.git
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the Flask application
``` python app.py ```

4. Open your browser and visit

``` http://127.0.0.1:5000/ ```

# Flask Web Application
The Flask app provides a simple and interactive user interface where users can:
- Enter flower measurements
- Submit data
- Receive real-time predictions of Iris species

# Results
- The Logistic Regression model achieved 100% accuracy
- Confusion matrix shows perfect classification
- Real-time predictions through Flask UI

# Future Enhancements
- Deploy application on cloud platforms (AWS/Render)
- Compare with other ML models (SVM, Random Forest)
- Add REST API support
- Improve frontend UI
- Display prediction probabilities graphically

# References
- UCI Machine Learning Repository – Iris Dataset
- Scikit-learn Documentation
- Flask Official Documentation

**Author**

Shruti Khadtar

Information Technology | Machine Learning | Data Analytics
