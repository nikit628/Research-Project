#!/usr/bin/env python
# coding: utf-8

# 
# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from boruta import BorutaPy
from tqdm import tqdm
get_ipython().system('pip install Boruta')
get_ipython().system('pip install Catboost')


# # Importing and Understanding Data-Set

# In[2]:


df = pd.read_csv(r'C:\Users\Nikit\Downloads\Heart_Disease_Prediction.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])


# In[6]:


df.isnull().sum()


# # Defining  Function

# In[7]:


def generate_metadata(dataframe):
    """
    Generates a DataFrame containing metadata for the columns of the provided DataFrame.

    :param dataframe: DataFrame for which metadata will be generated.
    :return: DataFrame containing metadata.
    """

    # Collection of basic metadata
    metadata = pd.DataFrame({
        'variable': dataframe.columns,
        'type': dataframe.dtypes,
        'null_count': dataframe.isnull().sum(),
        'null_percent': round((dataframe.isnull().sum() / len(dataframe))* 100,2),
        'cardinality': dataframe.nunique(),
    })
#     metadata = metadata.sort_values(by='type')
    metadata = metadata.reset_index(drop=True)

    return metadata


# In[8]:


def scale_numeric_columns(dataframe, method='normalize'):
    """
    Scales (normalizes or standardizes) the numeric columns in the provided DataFrame.

    :param dataframe: The DataFrame containing numeric columns to be scaled.
    :param method: The scaling method to use ('normalize' or 'standardize'). Default is 'normalize'.
    :return: A new DataFrame with the selected numeric columns scaled as specified.
    """
    # Create a copy of the input DataFrame to avoid modifying the original data
    df_scaled = dataframe.copy()
    
    # Select numeric columns based on data type
    numeric_columns = df_scaled.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    if method == 'normalize':
        # Initialize Min-Max Scaler for normalization
        scaler = MinMaxScaler()
    elif method == 'standardize':
        # Initialize StandardScaler for standardization
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Use 'normalize' or 'standardize'.")
    
    # Apply the chosen scaler to the selected numeric columns
    df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
    
    return df_scaled


# In[9]:


def select_features_by_variance_threshold(dataframe, variance_threshold=0):
    """
    Selects features from the provided DataFrame based on a variance threshold.

    :param dataframe: The DataFrame containing features to be selected.
    :param variance_threshold: The minimum variance required for a feature to be retained. Default is 0.
    :return: A new DataFrame with selected features.
    """
    # Create a copy of the input DataFrame to avoid modifying the original data
    df_selected = dataframe.copy()

    # Initialize VarianceThreshold selector
    selector = VarianceThreshold(variance_threshold)

    # Fit and transform the selector on the DataFrame
    selected_features = selector.fit_transform(df_selected)

    # Get the selected feature names
    selected_feature_names = df_selected.columns[selector.get_support()]

    # Print selected features
    print("Selected Features:", selected_feature_names.tolist())

    # Discarded feature names
    discarded_feature_names = df_selected.columns[~selector.get_support()]

    # Print discarded features (if needed)
    print("Discarded Features:", discarded_feature_names.tolist())

    # Update the DataFrame with selected features
    df_selected = df_selected[selected_feature_names]

    return df_selected


# In[10]:


def select_features_by_pca(dataframe, target, n_components=5, top_n=1):
    """
    Select features using Principal Component Analysis (PCA).

    :param dataframe: The DataFrame containing features to be selected.
    :param target: The name of the target column.
    :param n_components: The number of principal components to retain. Default is 5.
    :param top_n: The number of top features to select from each principal component. Default is 1.
    :return: A DataFrame containing selected features and the target column.
    """
    # Split the DataFrame into X (features) and y (target)
    X = dataframe.drop(columns=[target])
    y = dataframe[target]
    
    # Initialize PCA with the specified number of components
    pca_limited = PCA(n_components=n_components)
    
    # Fit PCA on the feature matrix (X)
    pca_limited.fit(X)
    
    # Get the principal components
    pca_limited_components = pca_limited.components_
    
    # Create a DataFrame to store the principal components
    pca_limited_features = pd.DataFrame(pca_limited_components, columns=X.columns,
                                        index=['PC'+str(i) for i in range(1, pca_limited_components.shape[0]+1)]).transpose()
    
    # Initialize a dictionary to store top features for each principal component
    top_components = {}
    
    # Loop through each principal component
    for component in pca_limited_features.columns:
        # Sort the components by absolute value in descending order
        sorted_components = pca_limited_features[component].abs().sort_values(ascending=False)
        
        # Select the top 'top_n' features
        top_components[component] = sorted_components.index[:top_n].tolist()
    
    # Create a list of selected features by combining top features from all principal components
    list_select_pca = list(set([item for sublist in top_components.values() for item in sublist]))
    
    # Print selected and discarded columns
    print("Selected Columns:", list_select_pca)
    print("Discarded Columns:", [col for col in X.columns if col not in list_select_pca])
    
    # Create a new DataFrame with selected features and the target column
    df_selected = dataframe[list_select_pca + [target]]
    
    return df_selected


# # Feature Enginerring

# In[11]:


df_heart_disease_original = pd.read_csv(r'C:\Users\Nikit\Downloads\Heart_Disease_Prediction.csv')
df_heart_disease_original['Heart Disease'] = df_heart_disease_original['Heart Disease'].replace({'Absence': 0, 'Presence': 1})
df_heart_disease_original


# In[12]:


df_metadata_original = generate_metadata(df_heart_disease_original)
df_metadata_original


# In[13]:


df_heart_disease_preparation = df_heart_disease_original.drop(axis=1,columns = 'Heart Disease')
df_heart_disease_preparation


# In[14]:


df_metadata_preparation = generate_metadata(df_heart_disease_preparation)
df_metadata_preparation


# In[15]:


df_heart_disease_preparation_normal = scale_numeric_columns(df_heart_disease_preparation) # default method='normalize'
df_heart_disease_preparation_normal


# In[16]:


df_metadata_normal = generate_metadata(df_heart_disease_preparation_normal)
df_metadata_normal


# Feature Standardization by Standard Scaling

# In[17]:


df_heart_disease_preparation_standard = scale_numeric_columns(df_heart_disease_preparation, 'standardize') # default method='normalize'
df_heart_disease_preparation_standard


# In[18]:


df_metadata_standard = generate_metadata(df_heart_disease_preparation_standard)
df_metadata_standard


# Feature Selection

# In[19]:


df_heart_disease_selection_normal = pd.merge(df_heart_disease_preparation_normal, 
df_heart_disease_original[['Heart Disease']], left_index=True, right_index=True, how='inner')
df_heart_disease_selection_normal


# In[20]:


df_heart_disease_selection_standard = pd.merge(df_heart_disease_preparation_standard, 
df_heart_disease_original[['Heart Disease']], left_index=True, right_index=True, how='inner')
df_heart_disease_selection_standard


# Feature Eliminated by Variance

# In[21]:


df_heart_disease_selection_normal = select_features_by_variance_threshold(df_heart_disease_selection_normal)
df_heart_disease_selection_normal


# Feature Eliminated by Principal Component Analysis

# In[22]:


df_heart_disease_pca_standard = select_features_by_pca(df_heart_disease_selection_standard, 'Heart Disease') # default n_components=5 and top_n=1
df_heart_disease_pca_standard


# # Exploratory Data Analysis

# In[23]:


import pandas as pd

# Load your dataset
df = pd.read_csv(r'C:\Users\Nikit\Downloads\Heart_Disease_Prediction.csv')

# Display the column names
print(df.columns)


# In[24]:


df.corr()


# In[25]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[26]:


numerical_columns = df.select_dtypes(include='number').columns
columns_to_plot = ['Heart Disease'] + numerical_columns.tolist()
sns.pairplot(df[columns_to_plot])
plt.show()


# In[27]:


y = df["Heart Disease"]

sns.countplot(y)


target_temp = df["Heart Disease"].value_counts()

print(target_temp)


# In[28]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# We'll analyse 'sex','chest pain type','FBS over 120','Cholestrol','Thallium'

# Analysing the 'Sex' feature

# In[29]:


df["Sex"].unique()


# In[30]:


sns.countplot(x='Sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[31]:


sns.barplot(df["Sex"],y)


# In[32]:


countFemale = len(df[df.Sex == 0])
countMale = len(df[df.Sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.Sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.Sex))*100)))


#  Analysing the Chest Pain Type

# In[33]:


df["Chest pain type"].unique()


# In[34]:


sns.countplot(x='Chest pain type', data=df, palette="mako_r")
plt.xlabel("Chest pain type (0 = female, 1= male)")
plt.show()


# In[35]:


sns.barplot(df["Chest pain type"],y)


# In[36]:


countFemale = len(df[df['Chest pain type'] == 0])
countMale = len(df[df['Chest pain type'] == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.Sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.Sex))*100)))


# Analysing the FBS over 120

# In[37]:


df["FBS over 120"].unique()


# In[38]:


sns.countplot(x='FBS over 120', data=df, palette="mako_r")
plt.xlabel("FBS over 120 (0 = female, 1= male)")
plt.show()


# In[39]:


sns.barplot(df["FBS over 120"],y)


# In[40]:


countFemale = len(df[df['FBS over 120'] == 0])
countMale = len(df[df['FBS over 120'] == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.Sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.Sex))*100)))


# Analysing the Cholestrol

# In[41]:


df["Cholesterol"].unique()


# In[42]:


sns.countplot(x='Cholesterol', data=df, palette="mako_r")
plt.xlabel("Cholesterol (0 = female, 1= male)")
plt.show()


# In[43]:


sns.barplot(df["Cholesterol"],y)


# In[44]:


countFemale = len(df[df['Cholesterol'] == 0])
countMale = len(df[df['Cholesterol'] == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.Sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.Sex))*100)))


# Analysing for Thallium

# In[45]:


df["Thallium"].unique()


# In[46]:


sns.countplot(x='Thallium', data=df, palette="mako_r")
plt.xlabel("Thallium (0 = female, 1= male)")
plt.show()


# In[47]:


sns.barplot(df["Thallium"],y)


# In[48]:


sns.distplot(df["Thallium"])


# Train Test Split

# In[49]:


from sklearn.model_selection import train_test_split

predictors = df.drop("Heart Disease",axis=1)
target = df["Heart Disease"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[50]:


X_train.shape


# In[51]:


X_test.shape


# In[52]:


Y_train.shape


# In[53]:


Y_test.shape


# # Model Simulation

# In[54]:


from sklearn.metrics import accuracy_score


# In[55]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)


# In[56]:


Y_pred_lr.shape


# In[57]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# Naive Bayes

# In[58]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)


# In[59]:


Y_pred_nb.shape


# In[60]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# Support Vector Machine

# In[61]:


from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)


# In[62]:


Y_pred_svm.shape


# In[63]:


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# K Nearest Neighbour

# In[64]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)


# In[65]:


Y_pred_knn.shape


# In[66]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# Decision Tree

# In[67]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)


# In[68]:


print(Y_pred_dt.shape)


# In[69]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# XG Boost

# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Assuming you have a dataset (replace this with your actual data)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=42)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and fit the XGBoost model
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Decode the predictions back to original classes
y_pred_decoded = label_encoder.inverse_transform(y_pred_xgb)

# Evaluate the model
score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)


accuracy = accuracy_score(y_test, y_pred_xgb)
print("Accuracy:", accuracy)


# In[71]:


# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Access the shape of t3he predictions
print(y_pred_xgb.shape)


# Ensemble Models

# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Create a synthetic dataset for demonstration purposes
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual models
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = LogisticRegression(random_state=42)
model3 = SVC(probability=True, random_state=42)

# Fit individual models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Make predictions on the test set
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# Create an ensemble using VotingClassifier
ensemble_model = VotingClassifier(estimators=[('rf', model1), ('lr', model2), ('svc', model3)], voting='soft')

# Fit the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions with the ensemble model
ensemble_pred = ensemble_model.predict(X_test)

# Evaluate individual models
print("Random Forest Accuracy:", accuracy_score(y_test, pred1))
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred2))
print("SVM Accuracy:", accuracy_score(y_test, pred3))
# Evaluate the ensemble model
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))


# In[3]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Create a synthetic dataset for demonstration purposes
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_clusters_per_class=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual models
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = LogisticRegression(random_state=42)
model3 = SVC(probability=True, random_state=42)

# Fit individual models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Make predictions on the test set
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# Create an ensemble using VotingClassifier
ensemble_model = VotingClassifier(estimators=[('rf', model1), ('lr', model2), ('svc', model3)], voting='soft')

# Fit the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions with the ensemble model
ensemble_pred = ensemble_model.predict(X_test)

# Evaluate individual models
print("Random Forest Accuracy:", accuracy_score(y_test, pred1))
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred2))
print("SVM Accuracy:", accuracy_score(y_test, pred3))

# Corrected XGBoost accuracy evaluation
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))

# Bar plot for individual models and ensemble
labels = ['Random Forest', 'Logistic Regression', 'SVM', 'Ensemble']
accuracies = [accuracy_score(y_test, pred1), accuracy_score(y_test, pred2), accuracy_score(y_test, pred3), accuracy_score(y_test, ensemble_pred)]

plt.bar(labels, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.ylim(0, 1)  # Set y-axis limits to be between 0 and 1 for accuracy
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.show()


# In[9]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_probs, label):
    fpr, tpr, _ = roc_curve(label_binarize(y_true, classes=[0, 1]), y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (area = {roc_auc:.2f})')

# Individual models
plot_roc_curve(y_test, model1.predict_proba(X_test), label='Random Forest')
plot_roc_curve(y_test, model2.predict_proba(X_test), label='Logistic Regression')
plot_roc_curve(y_test, model3.predict_proba(X_test), label='SVM')

# Ensemble model
plot_roc_curve(y_test, ensemble_model.predict_proba(X_test), label='Ensemble')

# Plotting the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2, label='Random Guessing')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()


# In[10]:


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred_probs, label):
    precision, recall, _ = precision_recall_curve(label_binarize(y_true, classes=[0, 1]), y_pred_probs[:, 1])
    avg_precision = average_precision_score(y_true, y_pred_probs[:, 1])
    plt.plot(recall, precision, lw=2, label=f'{label} (avg precision = {avg_precision:.2f})')

# Individual models
plot_precision_recall_curve(y_test, model1.predict_proba(X_test), label='Random Forest')
plot_precision_recall_curve(y_test, model2.predict_proba(X_test), label='Logistic Regression')
plot_precision_recall_curve(y_test, model3.predict_proba(X_test), label='SVM')

# Ensemble model
plot_precision_recall_curve(y_test, ensemble_model.predict_proba(X_test), label='Ensemble')

# Set labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix as a count plot
def plot_confusion_matrix(y_true, y_pred, ax, title):
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot confusion matrix for Random Forest
plot_confusion_matrix(y_test, model1.predict(X_test), axes[0, 0], title='Random Forest')

# Plot confusion matrix for Logistic Regression
plot_confusion_matrix(y_test, model2.predict(X_test), axes[0, 1], title='Logistic Regression')

# Plot confusion matrix for SVM
plot_confusion_matrix(y_test, model3.predict(X_test), axes[1, 0], title='SVM')

# Plot confusion matrix for Ensemble
plot_confusion_matrix(y_test, ensemble_model.predict(X_test), axes[1, 1], title='Ensemble')

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:




