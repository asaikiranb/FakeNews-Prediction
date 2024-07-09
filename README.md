# Fake News Detection with Logistic Regression

## Overview
This project aims to detect fake news using a machine learning model. The dataset includes various news articles labeled as either real or fake. We employ a Logistic Regression model to predict the authenticity of news articles based on their content.

## Project Structure
1. **Data Loading and Preprocessing**: 
    - Import necessary libraries.
    - Load the dataset from a specified URL.
    - Handle missing values by replacing them with empty strings.
    - Combine `title` and `author` columns to form a new `content` column.
    - Apply text preprocessing techniques such as stemming and removal of stopwords.

2. **Feature Extraction**:
    - Convert textual data into numerical data using TF-IDF Vectorizer.

3. **Model Training and Evaluation**:
    - Split the data into training and test sets.
    - Train a Logistic Regression model.
    - Evaluate the model's accuracy on both training and test data.

4. **Prediction**:
    - Use the trained model to predict the authenticity of news articles.

## Files
- `FakeNews_Prediction.ipynb`: Jupyter notebook containing the entire code for loading data, preprocessing, training, and evaluating the model.

## Libraries Used
- pandas
- numpy
- re
- nltk
- sklearn

## Instructions to Run the Project
1. **Install Required Libraries**:
    ```bash
    pip install pandas numpy nltk scikit-learn
    ```
2. **Download NLTK Stopwords**:
    ```python
    import nltk
    nltk.download('stopwords')
    ```
3. **Run the Jupyter Notebook**:
    - Open and run all cells in the `FakeNews_Prediction.ipynb` notebook.

## Results
- The model achieved an accuracy of approximately 98.66% on the training data and 97.91% on the test data.

## Conclusion
The Logistic Regression model effectively predicts whether a news article is real or fake based on its content, demonstrating high accuracy in both training and testing phases. This project can be further enhanced by exploring more sophisticated models and additional feature engineering techniques.
