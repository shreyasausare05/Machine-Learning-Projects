import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

def MarvellousRegression(DataPath):
    border = "-"*40
    # Step 1 : Load data

    print(border)
    print("Step 1 : Load data")
    print(border)

    df = pd.read_csv(DataPath)

    print(df.head())

    #Step 2 : Remove unwanted column

    print(border)
    print("Step 2 : Remove unwanted column")
    print(border)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(df.head())

    #Step 3 : Check missing values

    print(border)
    print("Step 3 : Check missing values")
    print(border)

    print("toatl missing values :")
    print(border)
    print(df.isnull().sum())
    print(border)

    # Step 4 : statistical summary

    print(border)
    print("Step 4 : statistical summary")
    print(border)

    print(df.describe())

    # step 5 : Correlation

    print(border)
    print("step 5 : Correlation")
    print(border)

    print(df.corr())

    #Step 6 : seprate Indepedent and depedent variables

    print(border)
    print("Step 6 : seprate Indepedent and depedent variables")
    print(border)

    X = df[["TV","radio","newspaper"]]
    Y =df["sales"]

    print("Independent variables :")
    print(X.head())

    print("Depedent variable :")
    print(Y.head())

    # Step 7 : split the dataset

    print(border)
    print("Step 7 : split the dataset")
    print(border)

    X_train,X_test,Y_train,Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Traning data :",X_train.shape)
    print("Testing data :",X_test.shape)

    #Step 8 : Create and train model

    print(border)
    print("Step 8 : Create and train model")
    print(border)

    model = LinearRegression()

    model = model.fit(X_train,Y_train)

    print("Model trained succesfully")


    # Step 9 : test the model

    print(border)
    print("Step 9 : test the model")
    print(border)

    Y_pred = model.predict(X_test)

    print("Expected ans :")
    print(Y_test[:3])

    print("Predicted ans :")
    print(Y_pred[:3])


    # Step 10 : Evalulate the model 

    print(border)
    print("Step 10 : Evalulate the model")
    print(border)

    MSE = mean_squared_error(Y_test,Y_pred)

    RMSE = np.sqrt(MSE)

    R2 = r2_score(Y_test,Y_pred)

    print("MSE :",MSE)
    print("RSME :",RMSE)
    print("R2 :",R2)

    #step 11 : Display Coeficiant

    print(border)
    print("step 11 : Display Coeficiant")
    print(border)

    print("TV Coeficiant :",model.coef_[0])

    print("Radio Coeficiant :",model.coef_[1])

    print("News paper Coeficiant :",model.coef_[2])

    print("Intercept :",model.intercept_)
    


def main():
    MarvellousRegression("Advertising.csv")

if __name__ == "__main__":
    main()