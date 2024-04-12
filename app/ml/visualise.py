import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Read Data from CSV
df = pd.read_csv('loan_data.csv')

# Step 2: Define Features and Target
X = df[['CreditScore', 'MonthsEmployed', 'InterestRate']]  # Features: select columns based on your preference
y = df["Default"]  # Target variable

# Step 3: Create and Train the Decision Tree Classifier (CART Algorithm) with limited max_depth
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)  # Using 'entropy' criterion for CART
clf.fit(X, y)

# Step 4: Plot the Decision Tree
plt.figure(figsize=(10, 5))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Default', 'Default'], rounded=True)
plt.show()
