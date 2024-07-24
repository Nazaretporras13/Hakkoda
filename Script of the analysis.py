# Exc1_Nazaret
## Intern Hakkoda

### Methodology 


import pandas as pd

df = pd.read_csv('/Users/nazaporras/Desktop/data1.csv')
df.info()
print(df.columns)

# Rename columns
df = df.rename(columns={
    'Your current year of Study': 'Year of Study',
    'What is your CGPA?': 'CGPA',
    'Do you have Depression?': 'Depression',
    'Do you have Anxiety?': 'Anxiety',
    'Do you have Panic attack?': 'Panic Attack',
    'Did you seek any specialist for a treatment?': 'Treatment',
    'Choose your gender': 'Gender'
})

print(df.columns)

# Convert Timestamp to datetime format
pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
df.info()

# Clean Year of Study
df['Year of Study'] = df['Year of Study'].str.replace(r'\bYear\b', '', regex=True, case=False)

# Remove rows with missing values
df_cleaned = df.dropna()
print(df_cleaned)
df_cleaned['Age'] = df_cleaned['Age'].astype(int)
print(df_cleaned)

# Convert selected columns to lowercase
columns_to_lower = ['Depression', 'Anxiety', 'Panic Attack', 'Treatment']
df_cleaned.info
df_cleaned.loc[:, columns_to_lower] = df_cleaned[columns_to_lower].apply(lambda x: x.str.lower())
print(df_cleaned)


file_path = '/Users/nazaporras/Desktop/df_cleaned.csv'
df_cleaned.to_csv(file_path, index=False)


## Analysis
### DEPRESSION-GENDER

from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

data = df_cleaned[['Gender', 'Depression']]

# Contingency table
contingency_table = pd.crosstab(df_cleaned['Gender'], df_cleaned['Depression'])
print(contingency_table)

total_counts = contingency_table.sum(axis=1)
contingency_table_percent = contingency_table.div(total_counts, axis=0) * 100

# Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi2: {chi2}")
print(f"p-value: {p}")
print(f"DF: {dof}")
print(f"expected: {expected}")

# Bar plot
ax = contingency_table_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Contingency Table of Gender vs Depression')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.xticks(rotation=0)  # Adjust x-axis label rotation
plt.legend(title='Depression')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center', color='white', fontsize=10)

plt.show()

### ANXIETY VS GENDER

data = df_cleaned[['Gender', 'Anxiety']]

# Contingency table
contingency_table = pd.crosstab(df_cleaned['Gender'], df_cleaned['Anxiety'])
print(contingency_table)

total_counts = contingency_table.sum(axis=1)
contingency_table_percent = contingency_table.div(total_counts, axis=0) * 100
print(contingency_table_percent)

# Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi2: {chi2}")
print(f"p-value: {p}")

# Bar plot
ax = contingency_table_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Contingency Table of Gender vs Anxiety')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.xticks(rotation=0)  # Adjust x-axis label rotation
plt.legend(title='Anxiety')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center', color='white', fontsize=10)

plt.show()

### Year of Study and Depression

data = df_cleaned[['Year of Study', 'Depression']]

# Contingency table
contingency_table = pd.crosstab(df_cleaned['Year of Study'], df_cleaned['Depression'])
print(contingency_table)

total_counts = contingency_table.sum(axis=1)
contingency_table_percent = contingency_table.div(total_counts, axis=0) * 100

# Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi2: {chi2}")
print(f"p-value: {p}")

# Bar plot
ax = contingency_table_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Contingency Table of Year of Study vs Depression')
plt.xlabel('Year of Study')
plt.ylabel('Percentage')
plt.xticks(rotation=0)  # Adjust x-axis label rotation
plt.legend(title='Depression')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center', color='white', fontsize=10)

plt.show()

### AGE-ANXIETY

data = df_cleaned[['Age', 'Anxiety']]

# Contingency table
contingency_table = pd.crosstab(df_cleaned['Age'], df_cleaned['Anxiety'])
print(contingency_table)

total_counts = contingency_table.sum(axis=1)
contingency_table_percent = contingency_table.div(total_counts, axis=0) * 100

# Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi2: {chi2}")
print(f"p-value: {p}")

# Bar plot
ax = contingency_table_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Contingency Table of Age vs Anxiety')
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.xticks(rotation=0)  # Adjust x-axis label rotation
plt.legend(title='Anxiety')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center', color='white', fontsize=10)

plt.show()

##Shapiro y Normalidad

from statsmodels.formula.api import ols
df_cleaned['Depression'] = df_cleaned['Depression'].map({'yes': 1, 'no': 0})
formula = 'Depression ~ C(Gender) + Age'
model = ols(formula, data=df_cleaned).fit()
print(model.summary())





