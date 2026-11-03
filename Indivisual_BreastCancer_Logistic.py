import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix




#Step 1 : load the data

df = pd.read_csv("breast_cancer.csv")

print ("Shape of dataset :",df.shape)

print("first few records :")
print(df.shape)


#Step 2 : separate features and labels

X = df.drop("target",axis=1)
Y= df["target"]

print("X shape :",X.shape)
print("Y shape :",Y.shape)

#Step 3 : Split dataset for training and testing

X_train,X_test,Y_train,Y_test= train_test_split(
                                                    X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42
                                                    )

#Step 4 ; scale the feature

scalar = StandardScaler()

X_train=scalar.fit_transform(X_train)
X_test=scalar.fit_transform(X_test)


#Step 5 : create model

model = LogisticRegression(max_iter=1000)

#Step 6 ; Train the model

model = model.fit(X_train,Y_train)


#Step 7 : test the model

Y_pred = model.predict(X_test)


#Step 8 : Evaluate model

print("Accuracy is :",accuracy_score(Y_test,Y_pred))

print("confusion matrix ;")
print(confusion_matrix(Y_test,Y_pred))