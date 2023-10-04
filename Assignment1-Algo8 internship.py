#!/usr/bin/env python
# coding: utf-8

# In[427]:


import warnings
warnings.filterwarnings('ignore')


# In[428]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Exploration :- 

# In[429]:


#Load the dataset into a dataframe


# In[430]:


df=pd.read_csv("bank-full.csv",sep=';')


# In[431]:


#Clean the dataset by Removing missing values and outliers


# In[432]:


df.head()


# In[433]:


df.info()


# In[434]:


df.describe()


# In[435]:


df.dtypes


# In[436]:


df.shape


# In[437]:


len(df)


# In[438]:


df['pdays'].unique()


# In[439]:


df.isnull().sum()


# In[440]:


#There are attribute which are categorical variable and are of type object
# Converting categorical variable into numerical value.


# In[441]:


df['job'] = df['job'].astype({'job':'category'})
df['marital'] = df['marital'].astype({'marital':'category'})
df['education'] = df['education'].astype({'education':'category'})
df['default'] = df['default'].astype({'default':'category'})
df['housing'] = df['housing'].astype({'housing':'category'})
df['loan'] = df['loan'].astype({'loan':'category'})
df['contact'] = df['contact'].astype({'contact':'category'})
df['month'] = df['month'].astype({'month':'category'})
df['poutcome'] = df['poutcome'].astype({'poutcome':'category'})
df['y'] = df['y'].astype({'y':'category'})


# In[442]:


df.info()


# In[443]:


df.describe(include='category')


# In[444]:


df['y'].value_counts(normalize=True)*100


# In[445]:


# percentage or each value type in each type of column 
for column in df:# or #for column in bankdata.columns:
    if df[column].dtype==object:    # check if column is object type or not 
        print(column,' :\n',df[column].value_counts()/df.shape[0]*100,sep='') # print unique value in each column (object type)
        print('-x-'*50)


# # Exploratory Data Analysis

# # Univariate Analysis

# # Q1. What is the distribution of the customer ages?

# In[446]:


sns.distplot(df['age'])


# In[447]:


plt.hist(df['age'], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Customer Ages')
plt.show()


# In[448]:


sns.histplot(df['age'], kde=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('KDE Plot of Customer Ages')
plt.show()


# In[409]:


sns.violinplot(x='age', data=df)
plt.xlabel('Age')
plt.title('Violin Plot of Customer Ages')
plt.show()


# In[449]:


sns.boxenplot(df['age'])


# In[450]:


sns.boxplot(df['duration'])


# In[451]:


sns.boxplot(df['campaign'])


# In[452]:


sns.boxplot(df['pdays'])


# In[453]:


sns.boxplot(df['previous'])


# In[454]:


sns.distplot(df['age'],bins=10, kde=False)


# In[455]:


#Frequency of 'subscribed'
df['y'].value_counts()


# In[456]:


# Plotting the 'subscribed' frequency
sns.countplot(data=df, x='y')


# In[457]:


#Normalizing the frequency table of 'Subscribed' variable
df['y'].value_counts(normalize=True)*100


# In[458]:


#Frequency table
df['job'].value_counts()


# In[459]:


# Plotting the job frequency table
sns.set_context('paper')
df['job'].value_counts().plot(kind='bar', figsize=(10,6));


# In[460]:


df['marital'].value_counts()


# In[461]:


sns.countplot(data=df, x='marital');


# In[462]:


sns.countplot(data=df, x='marital', hue='y');


# In[463]:


sns.distplot(df['age']);


# # Bivariate Analysis

# # Q2.) Relationship between customer age and subscription

# In[464]:


print(pd.crosstab(df['age'],df['y']))


# In[465]:


job = pd.crosstab(df['age'],df['y'])
job_norm = job.div(job.sum(1).astype(float), axis=0)


# In[466]:


job_norm.plot.bar(stacked=True,figsize=(8,6));


# # job and subscribed

# In[467]:


print(pd.crosstab(df['job'],df['y']))


# In[468]:


job = pd.crosstab(df['job'],df['y'])
job_norm = job.div(job.sum(1).astype(float), axis=0)


# In[469]:


job_norm.plot.bar(stacked=True,figsize=(8,6));


# # Marital status and subscribed

# In[470]:


#Marital status vs subscribed
pd.crosstab(df['marital'], df['y'])


# In[471]:


marital = pd.crosstab(df['marital'], df['y'])
marital_norm = marital.div(marital.sum(1).astype(float), axis=0)
marital_norm


# In[472]:


marital_norm.plot.bar(stacked=True, figsize=(10,6));


# # default and subscription

# In[473]:


pd.crosstab(df['y'],df['y'])


# In[474]:


dflt = pd.crosstab(df['default'], df['y'])
dflt_norm = dflt.div(dflt.sum(1).astype(float), axis=0)
dflt_norm


# In[475]:


dflt_norm.plot.bar(stacked=True, figsize=(6,6))


# # Converting the target variables into 0s and 1s

# In[476]:


# Converting the target variables into 0s and 1s
df['y'].replace('no', 0,inplace=True)
df['y'].replace('yes', 1,inplace=True)


# In[477]:


df['y']


# # Q3) Are there any other factors that are correlated with subscription?

# # Correlation matrix

# In[478]:


corr = df.corr()
corr


# In[479]:


fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corr, annot=True, cmap='viridis')


# #  Create Cross-Tabulations for Categorical Features:

# In[480]:


# Cross-tabulation for 'job' and subscription
job_crosstab = pd.crosstab(df['job'], df['y'])

# Cross-tabulation for 'marital' and subscription
marital_crosstab = pd.crosstab(df['marital'], df['y'])

# Print or visualize the cross-tabulations
print(job_crosstab)
print(marital_crosstab)


# # Use Chi-Squared Tests for Independence:

# In[481]:


from scipy.stats import chi2_contingency

# Perform chi-squared test for 'job' and subscription
chi2, p, _, _ = chi2_contingency(job_crosstab)

# Print the chi-squared statistic and p-value
print(f'Chi-Squared Statistic: {chi2}')
print(f'P-Value: {p}')


# # Assigning numbers to categories

# In[482]:


df['job']= df['job'].replace({'entrepreneur':1, 'management':2,'technician':3,'admin.':4,'services':5, 'self-employed':6,'blue-collar':7,'retired':8,'unemployed':9,'housemaid':10,'student':11,'unknown':12})
df['education'] = df['education'].replace({'primary':1,'secondary':2,'tertiary':3,'unknown':4})
df['housing'] = df['housing'].replace({'yes':1, 'no':0})
df['default'] = df['default'].replace({'yes':1, 'no':0})
df['loan'] = df['loan'].replace({'yes':1, 'no':0})
df['y'] = df['y'].replace({'yes':1, 'no':0})


# In[483]:


sns.barplot(x = 'y', y = 'age', data = df)


# In[484]:


sns.boxenplot(x='y', y=('duration'), data=df)


# In[485]:


sns.barplot(x='y', y='pdays', data=df)


# In[486]:


sns.barplot(x='y', y='duration', data=df)


# In[487]:


sns.barplot(x='y', y='previous', data=df)


# # Model Building

# In[488]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


# In[489]:


#Only the most relevant customer information is considered, which includes job title, education, age, 
#balance, default record, housing record and loan record. Other information, 
#such as ‘the number of contacts performed before this campaign’, 
#is omitted because it is not directly related to customers themselves.


# In[490]:


df = df.drop(columns=['marital', 'contact', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])


# In[491]:


#creting X and y variable 
X = df.drop('y', axis=1)
y = df['y']


# In[492]:


from sklearn.model_selection import train_test_split


# In[493]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state =1)


# # Logistic Regression

# In[494]:


# Create and fit the logistic regression model
LR = LogisticRegression()
LR.fit(X_train,y_train)
LR.score(X_train,y_train)


# In[495]:


LR_pred = LR.predict(X_test)


# In[496]:


print(classification_report(LR_pred,y_test))


# In[497]:


print("Accuracy:", accuracy_score(y_test, LR_pred))
print("F1-Score:", f1_score(y_test, LR_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, LR_pred))
print("Classification Report:\n", classification_report(y_test, LR_pred))


# # Q4. What is the accuracy of the logistic regression model?
Accuracy= Total Number of Predictions/Number of Correct Predictions
Accuracy is a measure of a model's overall correctness in making predictions.
It calculates the proportion of correctly classified instances (samples) out of the total instances in the dataset.
 
Accuracy is suitable for balanced datasets but can be misleading in imbalanced datasets.

The accuracy of the logistic Regression model is 89%
# # Q 5. What are the most important features for the logistic regression model? 

# In[498]:


# Get the coefficients of the logistic regression model
coefficients = LR.coef_[0]

# Create a DataFrame to associate feature names with coefficients
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

# Sort the DataFrame by absolute coefficient values in descending order
coefficients_df['Abs_Coefficient'] = abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

# Visualize the most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Abs_Coefficient', y='Feature', data=coefficients_df)
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.title('Most Important Features for Logistic Regression Model')
plt.show()


# # Q6. What is the precision of the logistic regression model?

# Precision:Precision measures the model's ability to correctly identify positive instances among the instances it predicts as positive (true positives) out of all predicted positive instances.
# It is especially relevant when there is a cost associated with false positives.
# Formula: Precision=True Positives/(True Positives + False Positives)
# 
# Precision for class 0 (negative class) is 1.00.

# # Q7. What is the recall of the logistic regression model?

# Recall (Sensitivity or True Positive Rate):Recall measures the model's ability to correctly identify all positive instances (true positives) out of all actual positive instances.
# It is especially relevant when missing positive cases is costly or critical.
# Formula: Recall=True Positives/(True Positives + False Negatives)
# 
# Recall for class 0 (negative class) is 0.89.
# 

# # Q8. What is the f1-score of the logistic regression model?

# F1-Score:The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
# It is a useful metric when you want to consider both false positives and false negatives.
# Formula: F1-Score=2*Precision*Recall/(Precision + Recall)
# 
# F1-score is particularly useful when dealing with imbalanced datasets.
# F1-score for class 0 (negative class) is 0.94.

# # Q9. How can you improve the performance of the logistic regression model?

# # To improve the performance of the logistic regression model, you can consider various techniques:
# 1) Feature engineering: Select and engineer features that are more informative.
# 2)Hyperparameter tuning: Optimize hyperparameters like regularization strength.
# 3)Handling class imbalance: If there's class imbalance in your data, consider techniques like oversampling or undersampling.
# 4)Trying different models: Experiment with other classification algorithms to see if they perform better.
# Here is the implementation-

# In[499]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Load your dataset (replace 'X' and 'y' with your features and target variable)
# X = df.drop('y', axis=1)
# y = df['y']

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardize numerical features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalanced data using Random Oversampling
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=1)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
lr = LogisticRegression(max_iter=1000, random_state=1)
grid_search = GridSearchCV(lr, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1), scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best hyperparameters
best_lr = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_lr.predict(X_test_scaled)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# # Q10. What are the limitations of the logistic regression model?

# # Limitations of the logistic regression model include:
# - Linearity assumption: Logistic regression assumes a linear relationship between the features and the log-odds of the response variable.
# - Limited expressiveness: Logistic regression may not capture complex relationships in the data.
# - Sensitive to outliers: Outliers can have a significant impact on logistic regression.
# - Assumption of independence: Logistic regression assumes that features are independent of each other, which may not hold in all cases

# # KNN Classifier

# In[500]:


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_train,y_train)


# In[501]:


knn.fit(X_test,y_test)
knn.score(X_test,y_test)


# In[502]:


knn_pred = knn.predict(X_test)


# In[503]:


print(classification_report(knn_pred,y_test))


# In[504]:


print("Accuracy:", accuracy_score(y_test, knn_pred))
print("F1-Score:", f1_score(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred))


# # Decision Tree

# In[505]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
DT_train_score= DT.score(X_train, y_train)
DT_train_score


# In[506]:


DT.fit(X_test,y_test)
DT_test_score = DT.score(X_test,y_test)
DT_test_score


# In[507]:


y_pred = DT.predict(X_test)


# In[508]:


print(classification_report(y_pred,y_test))


# In[509]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# # Random Forest

# In[510]:


RF = RandomForestClassifier()
RF.fit(X_train, y_train)


# In[511]:


RF_score_train = RF.score(X_train, y_train)
RF_score_train


# In[512]:


RF_score_test = RF.score(X_test, y_test)
RF_score_test


# In[513]:


pred_RF = RF.predict(X_test) 


# In[514]:


print (classification_report(y_test, pred_RF))


# In[515]:


print("Accuracy:", accuracy_score(y_test, pred_RF))
print("F1-Score:", f1_score(y_test, pred_RF))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_RF))
print("Classification Report:\n", classification_report(y_test, pred_RF))


# In[516]:


rfclass = RandomForestClassifier(n_estimators = 50)
rfclass.fit(X_train, y_train)


# In[517]:


rfclass.fit(X_test, y_test)


# In[518]:


RF_score_train = rfclass.score(X_train, y_train)
RF_score_train


# In[519]:


RF_score_test = rfclass.score(X_test, y_test)
RF_score_test


# In[520]:


rf_pred = rfclass.predict(X_test)


# In[521]:


print(classification_report(y_test,rf_pred))


# In[522]:


print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1-Score:", f1_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))


# # Adaboost Classifier

# In[523]:


adb = AdaBoostClassifier(n_estimators= 50, learning_rate=1.0, random_state=12)
adb.fit(X_train,y_train)


# In[524]:


AD_score_train = adb.score(X_train,y_train)
AD_score_train


# In[525]:


adb.fit(X_test,y_test)


# In[526]:


AD_score_test = adb.score(X_test,y_test)
AD_score_test


# In[527]:


adb_pred = adb.predict(X_test)


# In[528]:


print(classification_report(y_test,adb_pred))


# In[530]:


print("Accuracy:", accuracy_score(y_test, adb_pred))
print("F1-Score:", f1_score(y_test, adb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, adb_pred))
print("Classification Report:\n", classification_report(y_test, adb_pred))


# # Bagging

# In[531]:


bgg = BaggingClassifier(n_estimators= 50)
bgg.fit(X_train,y_train)


# In[532]:


Bagg_score_trn = bgg.score(X_train,y_train)
Bagg_score_trn


# In[533]:


bgg.fit(X_test,y_test)


# In[534]:


Bagg_score_tst = bgg.score(X_test,y_test)
Bagg_score_tst


# In[535]:


bagg_pred = bgg.predict(X_test)


# In[536]:


print(classification_report(bagg_pred,y_test))


# In[537]:


print("Accuracy:", accuracy_score(y_test, bagg_pred))
print("F1-Score:", f1_score(y_test, bagg_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, bagg_pred))
print("Classification Report:\n", classification_report(y_test, bagg_pred))


# # Model Evaluation

# In[538]:


d = {'Model': ['DTree', 'RF', 'AdaBoost', 'Bagging'], 
     'Training val':[Dtree_score_trn, RF_score_trn, AD_score_trn, Bagg_score_trn, ], 
     'Test Val': [Dtree_score_tst, RF_score_tst, AD_score_tst, Bagg_score_tst],}
print (d)


# In[539]:


m_eval = pd.DataFrame(d)
m_eval


# In[540]:


plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
plt.plot(m_eval['Model'], m_eval['Training val'], marker = '*')
plt.plot(m_eval['Model'], m_eval['Test Val'], marker = 'x')
plt.show()


# In[ ]:




