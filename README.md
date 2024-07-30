# Amazon Alexa Product Review Sentiment Analysis
This project involves analyzing Amazon Alexa product reviews to predict customer sentiment, categorizing feedback as either positive or negative. 
Utilizing a dataset with customer reviews, ratings, feedback types, and product variations, the goal is to preprocess and vectorize the text data, train several machine learning models, and select the best-performing model for sentiment classification.

![Image about the final project](<Amazon Alexa Reviews Sentiment Analysis.png>)

## Prerequisites
To run this project, ensure you have the following dependencies installed:

- Python 3.x
- pandas
- matplotlib
- seaborn
- wordcloud
- nltk
- scikit-learn
- xgboost
- joblib
- fastapi
- uvicorn

## Overview of the Code

1. Data Loading and Exploration
- Loads and explores the dataset to understand the distribution of ratings, feedback, and variations.
- Performs Exploratory Data Analysis (EDA) to visualize the data distribution and handle missing values.

2. Data Preprocessing
- Handles missing values and preprocesses text data by stemming and removing stop words.
- Vectorizes the text data using CountVectorizer to convert it into numerical format suitable for machine learning models.

3. Model Training and Evaluation
- Splits the data into training and testing sets and scales the features.
- Trains several machine learning models, including Logistic Regression, Linear SVC, XGBoost, Decision Tree, Random Forest, and Gradient Boosting.
- Evaluates the performance of each model using accuracy, confusion matrix, and classification report.

4. Model Selection and Saving
- Selects the best-performing model (Decision Tree Classifier) based on evaluation metrics.
- Saves the trained model, scaler, and vectorizer using joblib for future use.

5. Deployment
- Implements a FastAPI application with endpoints for the home page and sentiment prediction.
- The application allows users to input reviews, processes the text data, and provides sentiment predictions.

## Accuracy
The Decision Tree Classifier achieved an accuracy of 93% on the test data. 
This high accuracy reflects the model's effective performance in classifying Amazon Alexa product reviews as positive or negative.

## Contributions
Contributions are welcome! If you have ideas for improving the model, the application, or the code, feel free to submit a pull request or open an issue. Your feedback and contributions will be highly appreciated.
