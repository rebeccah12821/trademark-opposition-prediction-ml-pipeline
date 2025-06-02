# USPTO Trademark Opposition Outcome Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('uspto_oppositions.csv')
df['trademark_age'] = df['opposition_year'] - df['filing_year']
df['age_group'] = pd.cut(df['trademark_age'], bins=[0,10,20,30,40,50], labels=['(0, 10]','(10, 20]','(20, 30]','(30, 40]','(40, 50]'])

# Feature selection
features = [
    'trademark_age', 'opposition_basis_confusion', 'mark_type_0', 'mark_type_1',
    'mark_type_2', 'mark_type_3', 'mark_type_4', 'non_use_claim',
    'international_class_20', 'international_class_32'
]
X = df[features]
y = df['outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Ensemble (Random Forest + Gradient Boosting)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
ensemble.fit(X_train, y_train)
accuracy = ensemble.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.3f}")

# Feature importance plot
importances = ensemble.estimators_[0].feature_importances_
feature_names = X.columns
plt.figure(figsize=(12,6))
sns.barplot(x=importances, y=feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Age-Mark Type Success Rate Heatmap
heatmap_data = df.pivot_table(index='mark_type', columns='age_group', values='outcome', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=0.3, vmax=0.9)
plt.title('Opposition Success Rate by Mark Type and Age Group')
plt.xlabel('Trademark Age Group')
plt.ylabel('Mark Type')
plt.show()

