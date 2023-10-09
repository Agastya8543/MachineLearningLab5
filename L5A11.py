import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Read the data.
df = pd.read_excel("embeddingsdatasheet-1.xlsx")

def train_and_evaluate_mlp_classifier(X_train, X_test, y_train, y_test):
  # Create an MLP classifier.
  mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

  # Train the classifier.
  mlp_classifier.fit(X_train, y_train)

  # Make predictions on the test data.
  y_pred = mlp_classifier.predict(X_test)

  # Calculate the accuracy score on the test data.
  accuracy = accuracy_score(y_test, y_pred)

  return accuracy

# Split the data into training and test sets.
X = df.drop('Label',axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the MLP classifier.
accuracy = train_and_evaluate_mlp_classifier(X_train, X_test, y_train, y_test)

# Print the accuracy score.
print(f"Accuracy on test data: {accuracy}")
