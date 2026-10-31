
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("C:/Users/shiwa/OneDrive/Dokumen/Desktop/Student Performance in Exam(dataset)/StudentsPerformance.csv")

# Data Cleaning
print(data.head())
data.info()
print(data.isnull().sum())
print("Duplicates:", data.duplicated().sum())

data = data.drop_duplicates()
data = data.fillna(data.mode().iloc[0])

print(data.describe())

# EDA 1: Math Score Distribution
sns.histplot(data=data, x="math score", kde=True)
plt.title("Math Score Distribution")
plt.show()

# EDA 2: Average Subject Scores
data[["math score", "reading score", "writing score"]].mean().plot(kind="bar")
plt.title("Average Subject Scores")
plt.ylabel("Average Marks")
plt.show()

# EDA 3: Correlation Heatmap
sns.heatmap(
    data[["math score", "reading score", "writing score"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Between Scores")
plt.show()

# EDA 4: Test Preparation vs Math Score
sns.boxplot(
    data=data,
    x="test preparation course",
    y="math score"
)
plt.title("Test Preparation vs Math Score")
plt.show()

# EDA 5: Gender vs Average Math Score
sns.barplot(
    data=data,
    x="gender",
    y="math score",
    errorbar=None
)
plt.title("Gender vs Average Math Score")
plt.show()

# Classification Target
# 1 = Math score below 40
# 0 = Math score 40 or above
data["Result"] = (data["math score"] < 40).astype(int)

X = data[["gender", "lunch", "test preparation course"]]
y = data["Result"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 1. Logistic Regression
m1 = LogisticRegression()
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", round(accuracy_score(y_test, p1) * 100, 2), "%")
print("Classification Report:\n",
      classification_report(y_test, p1, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, p1))

# 2. KNN
m2 = KNeighborsClassifier(n_neighbors=5)
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)

print("\n===== KNN =====")
print("Accuracy:", round(accuracy_score(y_test, p2) * 100, 2), "%")
print("Classification Report:\n",
      classification_report(y_test, p2, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, p2))

# 3. Naive Bayes
m3 = GaussianNB()
m3.fit(X_train, y_train)
p3 = m3.predict(X_test)

print("\n===== NAIVE BAYES =====")
print("Accuracy:", round(accuracy_score(y_test, p3) * 100, 2), "%")
print("Classification Report:\n",
      classification_report(y_test, p3, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, p3))

# 4. Decision Tree
m4 = DecisionTreeClassifier(random_state=42)
m4.fit(X_train, y_train)
p4 = m4.predict(X_test)

print("\n===== DECISION TREE =====")
print("Accuracy:", round(accuracy_score(y_test, p4) * 100, 2), "%")
print("Classification Report:\n",
      classification_report(y_test, p4, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, p4))

# 5. Random Forest
m5 = RandomForestClassifier(random_state=42)
m5.fit(X_train, y_train)
p5 = m5.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", round(accuracy_score(y_test, p5) * 100, 2), "%")
print("Classification Report:\n",
      classification_report(y_test, p5, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, p5))

# Confusion Matrix Heatmap: Logistic Regression
sns.heatmap(confusion_matrix(y_test, p1), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix Heatmap: KNN
sns.heatmap(confusion_matrix(y_test, p2), annot=True, fmt="d", cmap="Blues")
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix Heatmap: Naive Bayes
sns.heatmap(confusion_matrix(y_test, p3), annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix Heatmap: Decision Tree
sns.heatmap(confusion_matrix(y_test, p4), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix Heatmap: Random Forest
sns.heatmap(confusion_matrix(y_test, p5), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
