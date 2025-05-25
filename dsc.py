import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("student-mat.csv", delimiter=';')
print(df.shape)
# Display the first few rows of the DataFrame as a table
print(tabulate(df.head( ) , headers='keys' , tablefmt= 'pretty' ) )
# Display the first 8 columns of the DataFrame as a table
df_subset = df.iloc[:, :8]
print(tabulate(df_subset.head(8), headers='keys' ,tablefmt= 'pretty' ) )


target_cnt=df["age"].value_counts()
print (target_cnt)
sns.countplot(x='age', data=df).set_title("Distribution of target variables")
plt.show()


df.hist (figsize=(6, 14), layout=(4,4), sharex=False)
#Show the plot
plt.show()


#Boxplot
df.plot(kind='box', figsize=(15, 12), layout=(4, 4), sharex=False, subplots=True)
#display
plt.show()


#PLOTLY
#Mapping dictionary for Fedu values
fedu_mapping = {
    0: 'None',
    1: 'Primary Education (4th grade)',
    2: '5th to 9th grade',
    3: 'Secondary Education',
    4: 'Higher Education'
}

#Map numeric values to their corresponding descriptions
df['Fedu'] = df['Fedu'].map(fedu_mapping)

#Create a pie chart using plotly.express
fig = px.pie (df, names='Fedu', title='Education Received by Father', color_discrete_sequence=px.colors.qualitative.T10)

#Show the plot
fig.show()


#PLOTLY
#Mapping dictionary for Medu values

medu_mapping = {
    0: 'None',
    1: 'Primary Education (4th grade)',
    2: '5th to 9th grade',
    3: 'Secondary Education',
    4: 'Higher Education'
}
#Map numeric values to their corresponding descriptions
df['Medu'] = df['Medu'].map(medu_mapping)

#Create a pie chart using plotly.express
fig = px.pie(df, names ='Medu', title='Education Received by Mother', color_discrete_sequence=px.colors.qualitative.T10)

#Show the plot
fig.show()


#PLOTLY
#Mapping dictionary for studytime values

studytime_mapping = {
    1: '<2 hours',
    2: '2 to 5 hours',
    3: '5 to 10 hours',
    4: '>10 hours'
}

#Map numeric values to their corresponding descriptions
df['studytime'] = df ['studytime'].map(studytime_mapping)

#Create a pie chart using plotly.express
fig = px.pie(df, names='studytime', title='Weekly Study Time', color_discrete_sequence=px.colors.qualitative.T10)

#Show the plot
fig.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("student-mat.csv", delimiter=';')

studytime_mapping = {
    1: '<2 hours',
    2: '2 to 5 hours',
    3: '5 to 10 hours',
    4: '>10 hours'
}
#Map numeric values to their corresponding descriptions
df['studytime'] = df['studytime'].map(studytime_mapping)

#Create a countplot
plt.figure(figsize=(10, 5))
sns.countplot(x='studytime', hue='sex', data=df, palette ='seismic')

#Display the plot
plt.title('Weekly Study Time by Gender')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tabulate import tabulate

# Load dataset
df = pd.read_csv("student-mat.csv", delimiter=';')

# Mapping dictionary for failures values
failures_mapping = {
    0: 'No failures',
    1: '1 failure',
    2: '2 failures',
    3: '3 or more failures'
}

# Map numeric values to their corresponding descriptions
df['failures_labeled'] = df['failures'].map(failures_mapping)

# Create a countplot
plt.figure(figsize=(10, 5))
sns.countplot(x='failures_labeled', hue='sex', data=df, palette='seismic')
plt.title('Number of Failures by Gender')
plt.xlabel('Failures')
plt.ylabel('Count')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("student-mat.csv", delimiter=';')

#Mapping dictionary for studytime values
studytime_mapping ={
    1: '<2 hours',
    2: '12 to 5 hours',
    3: '15 to 10 hours',
    4: '>10 hours'
}
df['studytime'] =df['studytime'].map(studytime_mapping)

#Mapping dictionary for failures values
failures_mapping ={
    0: 'No failures',
    1: '1 failure',
    2: '2 failures',
    3: '3 or more failures!'
}
#Map numeric values to their corresponding descriptions
df['failures'] =df['failures'].map(failures_mapping)

#Create a countplot
plt.figure(figsize=(12, 6))
sns.countplot(x='studytime', hue='failures', data= df, palette ='viridis')

#Display the plot
plt.title('Study Time vs Failures')
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("student-mat.csv", delimiter=';')

#Mapping dictionary for reason value
reason_mapping = {
    'home': 'Close to home',
    'reputation': 'School Roputation',
    'course': 'Course Reference',
    'other': 'Other'
}
#Map nominal values to their corresponding descriptions
df['reason']= df['reason'].map(reason_mapping)

#Create a countplot
plt.figure(figsize=(12, 6))
sns.countplot(x='school', hue='reason', data=df, palette='Set2')

#Display the plot
plt.title('School vs Reason for Choosing School')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("student-mat.csv", delimiter=';')

# Mapping dictionary for reason values
reason_mapping = {
    'home': 'Close to Home',
    'reputation': 'School Reputation',
    'course': 'Course Preference',
    'other': 'Other'
}

# Map reason codes to descriptive labels
df['reason'] = df['reason'].map(reason_mapping)

# Create a countplot
plt.figure(figsize=(12, 6))
sns.countplot(x='sex', hue='reason', data=df, palette='Set2')

# Display the plot
plt.title('Sex vs Reason for Choosing School')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Reason')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("student-mat.csv", delimiter=';')

#Selecting specific attributes for the pairplot
selected_attributes = ['studytime', 'traveltime', 'failures', 'age', 'sex']

#Create a pairplot for the selected attributes
sns.pairplot(df[selected_attributes], hue='sex', palette='Set2')
plt.suptitle("Pairwise Plot of Selected Student Attributes", y=1.02)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("student-mat.csv", delimiter=';')

# Selecting specific attributes for the pairplot
selected_attributes = ['G1', 'G2','G3', 'sex']

# Create a pairplot for the selected attributes
sns.pairplot(df[selected_attributes], hue='sex', palette='Set2')
plt.suptitle("Pairwise Plot of Selected Student Attributes", y=1.02)
plt.show()

# Exclude non-numeric columns
numeric_df = df.select_dtypes(include='number')

#Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

#Plot the correlation matrix using a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# data preprocessing
# kmeans clustering

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("student-mat.csv", delimiter=';')

# Select attributes for clustering
selected_attributes = ['G1', 'G2', 'G3']
numeric_columns = df[selected_attributes]

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(numeric_columns)

# Number of clusters
num_clusters = 2

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(data_standardized)

# Visualize the clusters using seaborn pairplot
sns.pairplot(df, hue='cluster', palette='Set2', vars=selected_attributes)
plt.suptitle("Pairwise Plot of Clusters for Selected Attributes that Indicate Grading", y=1.02)
plt.show()

#hierarchial clustering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load your data
df = pd.read_csv("student-mat.csv", delimiter=';')

# Select specific attributes for clustering
selected_attributes = ['G1', 'G2', 'G3']
numeric_columns = df[selected_attributes]

# Calculate the linkage matrix using Ward's method
linkage_matrix = linkage(numeric_columns, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(
    linkage_matrix,
    orientation='top',
    labels=df.index,
    distance_sort='descending',
    show_leaf_counts=True
)
plt.title("Hierarchical Clustering Dendrogram for Selected Attributes")
plt.xlabel("Student Index")
plt.ylabel("Distance")
plt.show()

# Extract clusters using a chosen distance threshold
distance_threshold = 30  # You can adjust this value based on dendrogram
df['cluster'] = fcluster(linkage_matrix, distance_threshold, criterion='distance')

# Visualize the clusters
sns.pairplot(df, hue='cluster', palette='Set2', vars=selected_attributes)
plt.suptitle("Pairwise Plot of Hierarchical Clusters for Selected Attributes", y=1.02)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Load dataset
df = pd.read_csv('student-mat.csv', delimiter=';')

# Convert categorical variables to numeric
binary_mappings = {
    'sex': {'M': 0, 'F': 1},
    'school': {'GP': 0, 'MS': 1},
    'address': {'U': 0, 'R': 1},
    'famsize': {'LE3': 0, 'GT3': 1},
    'Pstatus': {'T': 0, 'A': 1},
    'schoolsup': {'yes': 1, 'no': 0},
    'famsup': {'yes': 1, 'no': 0},
    'paid': {'yes': 1, 'no': 0},
    'activities': {'yes': 1, 'no': 0},
    'nursery': {'yes': 1, 'no': 0},
    'higher': {'yes': 1, 'no': 0},
    'internet': {'yes': 1, 'no': 0},
    'romantic': {'yes': 1, 'no': 0},
    'guardian': {'mother': 0, 'father': 1, 'other': 2}
}

for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

# Define features and target
predictors = df.iloc[:, :8].values
target = df.iloc[:, 32].values  # G3 column (final grade)

# Train-test split
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.25, random_state=42)

# Train Decision Tree
classifier = DecisionTreeClassifier(criterion='entropy', random_state=1, splitter='best')
classifier.fit(pred_train, tar_train)

# Predictions
predictions = classifier.predict(pred_test)

# Evaluation
print("Accuracy of training dataset is: {:.2f}".format(classifier.score(pred_train, tar_train)))
print("Accuracy of test dataset is: {:.2f}".format(classifier.score(pred_test, tar_test)))
print("Error rate is:", 1 - accuracy_score(tar_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(tar_test, predictions))

# Sensitivity (Recall for positive class)
print("Sensitivity is:", recall_score(tar_test, predictions, average='micro'))
# Specificity (1 - recall for negative class – simplified)
print("Specificity is:", 1 - recall_score(tar_test, predictions, average='micro'))

# Plot the decision tree
plt.figure(figsize=(20,10))
features = list(df.columns[:8])
plot_tree(classifier, feature_names=features, class_names=None, filled=True, rounded=True)
plt.show()

import time
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsOneClassifier
from sklearn.exceptions import UndefinedMetricWarning

# Load dataset
math_data = pd.read_csv('student-mat.csv', delimiter=';')

# Convert categorical variables using one-hot encoding
math_data = pd.get_dummies(math_data, columns=[
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
])

# Selected features
selected_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                     'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                     'Walc', 'health', 'absences', 'G1', 'G2']

X_mat = math_data[selected_features]  # Feature matrix
y_mat = math_data['G3']               # Target

# Split data
X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(
    X_mat, y_mat, test_size=0.2, random_state=42
)

# Train One-vs-One SVM Classifier
svm_classifier_mat = OneVsOneClassifier(SVC(kernel='linear', probability=True))
svm_classifier_mat.fit(X_train_mat, y_train_mat)

# Make predictions
predictions_mat = svm_classifier_mat.predict(X_test_mat)

# Evaluate performance
accuracy_mat = accuracy_score(y_test_mat, predictions_mat)
print("Accuracy for Mathematics dataset:", accuracy_mat)

# Confusion Matrix and Classification Report
conf_mat = confusion_matrix(y_test_mat, predictions_mat)
print("Confusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", classification_report(y_test_mat, predictions_mat))

# ROC Curve and AUC (micro-average)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    y_test_bin = label_binarize(y_test_mat, classes=np.unique(y_mat))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(np.unique(y_mat))):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], svm_classifier_mat.decision_function(X_test_mat)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(
    y_test_bin.ravel(), svm_classifier_mat.decision_function(X_test_mat).ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Mathematics dataset')
plt.legend(loc="lower right")
plt.show()

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
import sklearn.metrics

# Load the dataset
df = pd.read_csv("student-mat.csv", delimiter=';')

# Convert categorical variables to numerical
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1, 'other':2})

# Define predictor and target columns
# Instead of removing 'G3', define predictor columns as all columns except 'G3'
predictor_columns = df.columns.tolist()
predictor_columns.remove('G3') # Remove the target column

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['school', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                                 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                                 'internet', 'romantic'])

#After one-hot encoding, update predictor_columns to include new columns
predictor_columns = df.columns.tolist()
predictor_columns.remove('G3') #Remove target column again


predictors = df[predictor_columns].values
targets = df['G3'].values  # or use classification e.g., pass/fail

# Split the data into training and testing sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=0.25)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')
knn.fit(pred_train, tar_train)

# Make predictions
y_pred = knn.predict(pred_test)

# Evaluation
print("Accuracy is:", accuracy_score(tar_test, y_pred, normalize=True))
print("Classification error is:", 1 - accuracy_score(tar_test, y_pred, normalize=True))
print("Sensitivity is:", sklearn.metrics.recall_score(tar_test, y_pred, average='micro'))
print("Specificity is:", 1 - sklearn.metrics.recall_score(tar_test, y_pred, average='micro'))
