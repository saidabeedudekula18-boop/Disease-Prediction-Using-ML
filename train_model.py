import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("dataset/Training.csv")

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 133"], errors="ignore")

# Features and Target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

best_model = None
best_accuracy = 0

print("=" * 50)
print("Disease Prediction Model Comparison")
print("=" * 50)

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name:<20}: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save Best Model
joblib.dump(best_model, "models/disease_model.pkl")

print("\n" + "=" * 50)
print("Best Model Saved Successfully!")
print(f"Best Accuracy : {best_accuracy * 100:.2f}%")
print("=" * 50)