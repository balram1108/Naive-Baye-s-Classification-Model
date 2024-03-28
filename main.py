import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
import pandas as pd


df = pd.read_csv("/Users/jimmypostwala/Desktop/git_projects/naive_bayes/seasons_swimming_advisability_1000_rows.csv")

encoder = LabelEncoder()
df_encoded = df.apply(lambda col: encoder.fit_transform(col) if col.dtype == 'object' else col)

# Splitting the dataset into features (X) and target variable (y)
X = df_encoded.drop('CanSwim', axis=1)  # Features are all columns except 'CanSwim'
y = df_encoded['CanSwim']  # Target variable is 'CanSwim'

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the Naive Bayes classifier
model = CategoricalNB()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
