# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:23:03 2023
@author: Zahid Hasan
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords  # Import stopwords from nltk.corpus
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import gensim

nltk.download('stopwords')
nltk.download('punkt')

# Load the trained Random Forest model
model_path = 'trained_model.pkl'
RF_Model = joblib.load(model_path)

# Load stopwords and define function to remove stopwords and short words
nltk.download("stopwords")
stop_words = stopwords.words('english')

def remove_stop_words(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return ' '.join(result)

# Vectorizer 
vectorizer = CountVectorizer()

# Input directory containing resumes to be categorized
input_directory = r'C:\Users\Zahid Hasan\Downloads\resume classification\Resume'

# Output directory for categorized resumes
output_directory = r'C:\path\to\output\categorized_resumes'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the list of categories
categories = RF_Model.classes_

# Create a CSV file to store categorized resumes
csv_file = os.path.join(output_directory, 'categorized_resumes.csv')
categorized_data = {'filename': [], 'category': []}

for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_directory, filename)
        with open(file_path, 'r') as file:
            resume_text = file.read()

        cleaned_text = remove_stop_words(resume_text)
        vectorized_text = vectorizer.transform([cleaned_text]).astype(float)
        predicted_category = RF_Model.predict(vectorized_text)[0]

        # Move the resume file to the respective category folder
        category_folder = os.path.join(output_directory, predicted_category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        new_file_path = os.path.join(category_folder, filename)
        os.rename(file_path, new_file_path)

        # Add data to the CSV
        categorized_data['filename'].append(filename)
        categorized_data['category'].append(predicted_category)

# Write the categorized data to the CSV file
categorized_df = pd.DataFrame(categorized_data)
categorized_df.to_csv(csv_file, index=False)

print("Resumes categorized and moved to respective folders.")
