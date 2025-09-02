import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_samples = 500
data = {
    'CurrentRatio': np.random.uniform(0.5, 5, num_samples),
    'DebtEquityRatio': np.random.uniform(0, 2, num_samples),
    'ROA': np.random.uniform(-0.2, 0.3, num_samples),
    'Bankrupt': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]) # 20% bankruptcy rate
}
df = pd.DataFrame(data)
# Add some noise to make it more realistic.
df['CurrentRatio'] += np.random.normal(0, 0.5, num_samples)
df['DebtEquityRatio'] += np.random.normal(0, 0.2, num_samples)
df['ROA'] += np.random.normal(0, 0.05, num_samples)
# Ensure ratios are non-negative
df['CurrentRatio'] = df['CurrentRatio'].apply(lambda x: max(0, x))
df['DebtEquityRatio'] = df['DebtEquityRatio'].apply(lambda x: max(0, x))
# --- 2. Data Cleaning and Preparation ---
#Handle outliers (simple capping for demonstration)
for col in ['CurrentRatio', 'DebtEquityRatio', 'ROA']:
    upper_bound = df[col].quantile(0.95)
    lower_bound = df[col].quantile(0.05)
    df[col] = df[col].clip(lower_bound, upper_bound)
X = df[['CurrentRatio', 'DebtEquityRatio', 'ROA']]
y = df['Bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training and Evaluation ---
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Financial Ratios')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
plt.figure(figsize=(8,6))
sns.scatterplot(x='DebtEquityRatio', y='CurrentRatio', hue='Bankrupt', data=df)
plt.title('Debt-Equity Ratio vs. Current Ratio')
plt.savefig('scatter_plot.png')
print("Plot saved to scatter_plot.png")