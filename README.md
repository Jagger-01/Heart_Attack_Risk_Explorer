
# **Heart Attack Risk Explorer**
## **Introduction**

Understanding the factors that contribute to heart attacks is crucial in preventing and managing cardiovascular diseases. This project was inspired by a desire to delve into the complexities of heart health and to create a comprehensive dashboard that visualizes key risk factors associated with heart attacks. 

Using the [Heart Attack Risk Factors Dataset](https://www.kaggle.com/datasets/waqi786/heart-attack-dataset) sourced from Kaggle, I aimed to uncover meaningful insights through data analysis, feature engineering, and statistical testing. The final outcome is an interactive dashboard that provides a holistic view of heart attack risks and treatment effectiveness.

## **Project Workflow**

### **1. Data Acquisition**

The journey began with sourcing the dataset from Kaggle. The dataset, titled "Heart Attack Risk Factors Dataset," includes a variety of attributes related to patient demographics, medical history, and lifestyle factors. The goal was to use this data to identify critical risk factors for heart attacks and to visualize these insights in a user-friendly format.

### **2. Data Inspection**

Before diving into analysis, I conducted a thorough inspection of the dataset to ensure its quality. This step was crucial to identify any issues such as missing values, duplicates, outliers, and inconsistencies in data types.

Here’s the code snippet used for the inspection:

```python
import pandas as pd

# Load the dataset
file_path = '../Data/heart_attack_dataset.csv'
df = pd.read_csv(file_path)

# Summary of the dataset
print("Summary of the dataset:")
print(df.info())
print("\n")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())
print("\n")

# Check for duplicates
print("Number of duplicate rows in the dataset:")
print(df.duplicated().sum())
print("\n")

# Check for invalid or outlier values (e.g., negative ages, extremely high/low blood pressure)
print("Summary statistics for numerical columns:")
print(df.describe())
print("\n")

# Check for any unexpected values in categorical columns
print("Unique values in categorical columns:")
categorical_columns = ['Gender', 'Smoking Status', 'Chest Pain Type', 'Treatment']
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].unique())

# Check for any inconsistent data types
print("\nData types of each column:")
print(df.dtypes)
print("\n")
```

During this step, I confirmed that the dataset was clean with no major issues. However, there were opportunities for enhancing the data through feature engineering.

### **3. Feature Engineering**

To enrich the dataset and make the analysis more insightful, I introduced several new features. These features were designed to capture more nuanced aspects of heart health, such as categorizing patients into age groups, calculating risk scores, and determining the necessity for lifestyle modifications.

Here’s an example of the feature engineering code:

```python
import pandas as pd

# Load the dataset
file_path = '../Data/heart_attack_dataset.csv'
df = pd.read_csv(file_path)

# 1. Age Group
def age_group(age):
    if age < 20:
        return "<20"
    elif 20 <= age < 30:
        return "20-29"
    elif 30 <= age < 40:
        return "30-39"
    elif 40 <= age < 50:
        return "40-49"
    elif 50 <= age < 60:
        return "50-59"
    elif 60 <= age < 70:
        return "60-69"
    elif 70 <= age < 80:
        return "70-79"
    else:
        return "80+"

df['Age Group'] = df['Age'].apply(age_group)

# 2. Cholesterol Level Category
def cholesterol_category(cholesterol):
    if cholesterol < 200:
        return "Low"
    elif 200 <= cholesterol < 240:
        return "Borderline High"
    else:
        return "High"

df['Cholesterol Level Category'] = df['Cholesterol (mg/dL)'].apply(cholesterol_category)

# 3. Blood Pressure Category
def bp_category(bp):
    if bp < 120:
        return "Normal"
    elif 120 <= bp < 130:
        return "Elevated"
    elif 130 <= bp < 140:
        return "High Blood Pressure Stage 1"
    else:
        return "High Blood Pressure Stage 2"

df['Blood Pressure Category'] = df['Blood Pressure (mmHg)'].apply(bp_category)

# 4. Risk Score
def risk_score(row):
    score = 0
    if row['Cholesterol (mg/dL)'] > 240:
        score += 1
    if row['Blood Pressure (mmHg)'] >= 140:
        score += 1
    if row['Has Diabetes'] == 1:
        score += 1
    if row['Smoking Status'] == 'Current':
        score += 1
    return score

df['Risk Score'] = df.apply(risk_score, axis=1)

# 5. Heart Health Status
def heart_health_status(row):
    bp_cat = row['Blood Pressure Category']
    chol_cat = row['Cholesterol Level Category']
    if bp_cat == "Normal" and chol_cat == "Low":
        return "Healthy"
    elif bp_cat == "Elevated" or chol_cat == "Borderline High":
        return "At Risk"
    else:
        return "Unhealthy"

df['Heart Health Status'] = df.apply(heart_health_status, axis=1)

# 6. Lifestyle Modification Necessity
df['Lifestyle Modification Necessity'] = df.apply(lambda x: 'Yes' if x['Risk Score'] >= 2 or x['Smoking Status'] == 'Current' else 'No', axis=1)

# 7. Treatment Effectiveness Category
def treatment_effectiveness(row):
    if row['Chest Pain Type'] == 'Typical Angina' and row['Treatment'] == 'Lifestyle Changes':
        return "High"
    elif row['Chest Pain Type'] == 'Atypical Angina' and row['Treatment'] == 'Medication':
        return "Moderate"
    else:
        return "Low"

df['Treatment Effectiveness Category'] = df.apply(treatment_effectiveness, axis=1)

# 8. Create a Unique Identifier (Patient ID)
df['Patient ID'] = range(1, len(df) + 1)

# Save the updated dataframe to a new CSV file
output_path = '../Data/heart_attack_dataset_updated.csv'  # Update this path if needed
df.to_csv(output_path, index=False)

print("New columns added and saved to", output_path)
```

This step significantly improved the dataset’s usability for subsequent analysis and visualization.

### **4. Statistical Testing**

To ensure the features created were meaningful, I conducted statistical tests to identify significant correlations and relationships between variables. This helped to determine which factors were worth including in the final dashboard.

Here’s the code used for statistical testing:

```python
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# Load the updated dataset
file_path = '../Data/heart_attack_dataset_updated.csv'
df = pd.read_csv(file_path)

# Print each header and its unique values
print("Headers and their unique values:")
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].unique())

print("\n")

from scipy.stats import pearsonr, chi2_contingency, f_oneway, ttest_ind

# List of numerical and categorical columns
numerical_columns = ['Age', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)', 'Risk Score']
categorical_columns = ['Gender', 'Has Diabetes', 'Smoking Status', 'Chest Pain Type', 'Treatment',
                       'Age Group', 'Cholesterol Level Category', 'Blood Pressure Category',
                       'Heart Health Status', 'Lifestyle Modification Necessity', 'Treatment Effectiveness Category']

# Thresholds
correlation_threshold = 0.3  # for Pearson correlation
p_value_threshold = 0.05  # for Chi-square, ANOVA, and t-tests

# Results dictionaries
investigate = []
no_investigation = []

# 1. Numerical vs. Numerical (Correlation)
for i, col1 in enumerate(numerical_columns):
    for col2 in numerical_columns[i+1:]:
        corr, p_value = pearsonr(df[col1], df[col2])
        if abs(corr) > correlation_threshold:
            investigate.append(f"Correlation between {col1} and {col2}: corr = {corr:.2f}, p-value = {p_value:.4f}")
        else:
            no_investigation.append(f"Correlation between {col1} and {col2}: corr = {corr:.2f}, p-value = {p_value:.4f}")

# 2. Categorical vs. Categorical (Chi-Square Test)
for i, col1 in enumerate(categorical_columns):
    for col2 in categorical_columns[i+1:]:
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < p_value_threshold:
            investigate.append(f"Chi-Square test between {col1} and {col2}: p-value = {p_value:.4f}")
        else:
            no_investigation.append(f"Chi-Square test between {col1} and {col2}: p-value = {p_value:.4f}")

# 3. Numerical vs. Categorical (ANOVA/T-test)
for num_col in numerical_columns:
    for cat_col in categorical_columns:
        unique_categories = df[cat_col].unique()
        groups = [df[num_col][df[cat_col] == category] for category in unique_categories]
        if len(unique_categories) == 2:  # Use t-test for binary categories
            t_stat, p_value = ttest_ind(groups[0], groups[1])
        else:  # Use ANOVA for multiple categories
            f_stat, p_value = f_oneway(*groups)
        if p_value < p_value_threshold:
            investigate.append(f"Test between {num_col} and {cat_col}: p-value = {p_value:.4f}")
        else:
            no_investigation.append(f"Test between {num_col} and {cat_col}: p-value = {p_value:.4f}")

# Print the results
print("\n=== Should Investigate Further ===")
for item in investigate:
    print(item)

print("\n=== No Further Investigation Needed ===")
for item in no_investigation:
    print(item)
```

The results identified several key relationships worth exploring, such as the correlation between risk scores and blood pressure, and the impact of cholesterol levels on heart health status.

### **5. Dashboard Creation**

The **"Comprehensive Heart Disease Risk & Treatment Dashboard"** is designed to provide a clear view of heart attack risk factors, treatment outcomes, and the influence of lifestyle choices. You can view the dashboard [here](https://lookerstudio.google.com/reporting/d9824f95-a959-4536-a232-83931b40575f).

#### **Page 1: Heart Attack Risk Factors Overview**
This page offers a broad look at the key risk factors for heart attacks. The charts chosen highlight demographic data, risk scores, and correlations between cholesterol, blood pressure, and risk levels.

- **Demographic Distribution by Age Group and Gender**: Visualizes the patient demographics to identify high-risk groups.
- **Risk Score Distribution**: Shows the spread of risk scores across the population.
- **Cholesterol and Blood Pressure Correlation with Risk Score**: Explores the relationship between these critical health metrics and overall risk.
- **Lifestyle Modification Necessity by Smoking Status**: Illustrates the impact of smoking on the need for lifestyle changes.

#### **Page 2: Treatment and Health Outcomes**
This page focuses on analyzing the effectiveness of various treatments and their impact on heart health.

- **Treatment Effectiveness by Chest Pain Type**: Assesses how different treatments perform based on chest pain type.
- **Heart Health Status by Treatment Type**: Provides a snapshot of heart health outcomes by treatment.
- **Risk Score vs. Treatment Effectiveness**: Correlates risk scores with treatment efficacy.
- **Age Distribution by Treatment Type**: Displays how treatment varies across different age groups.

#### **Page 3: Lifestyle Factors and Heart Health**
This page examines how lifestyle factors like smoking and blood pressure affect heart health.

- **Impact of Smoking on Heart Health**: Highlights the detrimental effects of smoking on heart health.
- **Cholesterol Levels Across Age Groups**: Shows age-related trends in cholesterol levels.
- **Lifestyle Changes Needed by Blood Pressure Category**: Indicates which blood pressure categories require lifestyle changes.
- **Risk Scores Distribution by Lifestyle Modification Necessity**: Visualizes how risk scores relate to the need for lifestyle changes.

This structure provides a focused, data-driven approach to understanding heart disease risk factors and treatment outcomes.

### **Files in the Repository**

- **Data Folder**:
  - `heart_attack_dataset.csv`: The original dataset downloaded from Kaggle.
  - `heart_attack_dataset_updated.csv`: The dataset after feature engineering, ready for analysis and dashboard creation.

- **Notebooks Folder**:
  - `Feature_Engineering.ipynb`: Contains the code used to create new features such as Age Group, Risk Score, and Heart Health Status.
  - `Inspection.ipynb`: Contains the code used to inspect the dataset, check for missing values, duplicates, and outliers.
  - `Statistical_Testing.ipynb`: Contains the code used to perform statistical tests and identify significant relationships between variables.

- **Dashboard File**:
  - `Comprehensive_Heart_Disease_Risk_&_Treatment_Dashboard.pdf`: The final dashboard document, providing a comprehensive view of heart disease risk factors and treatment outcomes. [Here's a link to the dashboard itself.](https://lookerstudio.google.com/reporting/d9824f95-a959-4536-a232-83931b40575f)

### **Conclusion**

This project successfully identified key risk factors for heart attacks and visualized them in an interactive dashboard. The insights gained from this analysis can be used for further research or clinical decision-making to improve heart health outcomes.


### **Acknowledgments**

- Special thanks to the Kaggle community for providing the dataset.
- Appreciation to the developers of Python libraries such as pandas, scipy, and matplotlib, which were instrumental in this analysis.
