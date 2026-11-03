import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Step 1: Load Data
#--------------------------------------------------------------
#   function name : Loaddata
#   Description : Load the data from CSV
#   Input  :     Loan Default CSV file
#   Output :     Data frame
#   Author name :Shreyas Ausare
#   Date :      03/11/2026
#--------------------------------------------------------------

df = pd.read_csv('Loan_Default.csv')
print("Data Loaded Successfully")
print(df.head())
print(df.shape)


#Step 2: Preprocess Data
#--------------------------------------------------------------
#   function name : PreprocessData
#   Description : It perform data analysing
#   Input  :     Data frame
#   Output :     Updated Data frame
#   Author name :Shreyas Ausare
#   Date :      03/11/2026
#--------------------------------------------------------------


X = df.drop('Default', axis=1)
y = df['Default']

# Remove rows with missing target values
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

# Identify numerical and categorical columns

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Numerical preprocessing

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

#Step 3: Split Data
#--------------------------------------------------------------
#   function name : SplitData
#   Description : It splits the data into training and testing sets
#   Input  :     Data frame
#   Output :     Split data
#   Author name :Shreyas Ausare
#   Date :      03/11/2026
#--------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

#Step 4: Scale Data
#--------------------------------------------------------------
#   function name : ScaleData
#   Description : It scales the data
#   Input  :     Data frame
#   Output :     Scaled Data frame
#   Author name :Shreyas Ausare
#   Date :      03/11/2026
#--------------------------------------------------------------

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#Step 4: Train Model
#--------------------------------------------------------------
#   function name : Trained Model
#   Description : It perform Model training 
#   Input  :     Training ceatures  frame
#   Output :      Subset for trailing and testing
#   Author name :Shreyas Ausare
#   Date :      03/11/2026
#--------------------------------------------------------------

model_log=LogisticRegression()
model_log.fit(X_train, y_train)

model_rf=RandomForestClassifier()
model_rf.fit(X_train, y_train)

model_knn=KNeighborsClassifier()
model_knn.fit(X_train, y_train)

# Votting Classifier

voting_clf = VotingClassifier(
    estimators=[('lr', model_log),
                ('rf', model_rf), 
                ('knn', model_knn)],
                voting='soft')
voting_clf.fit(X_train, y_train)

print("Model Training Completed Successfully")


# Step 5 : Evaluate model

#-----------------------------------------------------
#   Function Name : EvaluateModel
#   Description :   It performs model testing
#   Input :         model, testing data (fetures ,labels)
#   Output :        none
#   Author :        Shreyas Ausare
#   Date :          03/11/2026
#-----------------------------------------------------

Y_pred = voting_clf.predict(X_test)
print("Voting Classifier Model Evaluation")

accuracy = accuracy_score(y_test, Y_pred)

print(f"Accuracy: {accuracy}")
