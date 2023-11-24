import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('../Data/training_data.csv').dropna()

X = data.drop('increase_stock', axis = 1)
y = data['increase_stock']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Hold-out validation

y_pred = model.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Cross validation
model = LogisticRegression(max_iter = 1000)

n_rows = len(X)

scores = cross_val_score(model, X, y, cv = 1000, scoring = 'accuracy')
print("Accuracy for each fold")
print(scores)

print("\nMean Accuracy: {:.2f}".format(scores.mean()))
print("Standard Deviation: {:.2f}".format(scores.std()))