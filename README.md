
# Sentiment Analysis on Animal Crossing Game Reviews


## Project Overview

This project focuses on building a sentiment analysis model that predicts the rating of an Animal Crossing game based on user reviews. By analyzing the sentiments expressed in the reviews, the model provides an accurate prediction of the rating users are likely to assign to the game.

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) was performed to understand the distribution and characteristics of the data. Key steps in the EDA process included:

- Data Exploration: Inspecting the overall structure and content of the data, including the distribution of ratings and the length of reviews.

### Data Preprocessing
A data preprocessing pipeline was implemented using a recipe in R to clean and prepare the data for modeling. The key steps included:

- Text Cleaning: Removal of unnecessary characters, links, and stop words from the reviews.
 - Tokenization: Breaking down the reviews into individual words (tokens) for analysis.
- Text Vectorization: Converting text data into numerical format using techniques like Term Frequency-Inverse Document Frequency (TF-IDF).

### Model Specification
The sentiment analysis model was specified using a regularized lasso logistic regression approach. 

Regularization was applied to prevent overfitting and to handle multicollinearity among the features.

### Model Tuning
Model tuning was performed to optimize the hyperparameters of the regularized logistic regression model. This involved:

- Grid Search: Exploring a range of hyperparameters to find the best combination for model performance.
 - Cross-Validation: Using k-fold cross-validation to evaluate the model's performance on different subsets of the data, ensuring generalizability.

### Model Evaluation
The final model was evaluated using a range of performance metrics, including Accuracy and ROC-AUC

The model achieved strong performance, demonstrating its capability to accurately predict user ratings based on review sentiment.
