from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset

run = Run.get_context()

def clean(data):
    # clean and one-hot encode data
    
    data = data.to_pandas_dataframe()

    # imputing missing values
    data['incomeVerified'].fillna(False,inplace=True)
    data['clientMaritalStatus'].fillna(data['clientMaritalStatus'].mode()[0],inplace=True)
    data['clientResidentialStauts'].fillna(data['clientResidentialStauts'].mode()[0],inplace=True)
    data['clientState'].fillna(data['clientState'].mode()[0],inplace=True)
    data['clientNumberPhoneCOntacts'].fillna(int(data['clientNumberPhoneCOntacts'].mean()),inplace=True)
    data['clientAvgCallsPerDay'].fillna(data['clientAvgCallsPerDay'].mean(),inplace=True)

    # one-hot encoding on categorical variables
    cat_df = ['clientGender','clientMaritalStatus', 'clientLoanPurpose','clientResidentialStauts','incomeVerified']
    data = pd.get_dummies(data,columns=cat_df,drop_first=True)

    data.drop(['loanId','dueDate','clientTimeAtEmployer','loanTerm','clientId'],axis=1,inplace=True)
    x_df = data.copy()
    y_df = x_df.pop("loanDefault")

    return x_df, y_df


# load dataset
url = "https://raw.githubusercontent.com/ChidiNdego/loan-default-capstone-project/master/starter_file/loan_default_prediction.csv"
data = TabularDatasetFactory.from_delimited_files(url)

x, y = clean(data)

# split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0, shuffle=True)

run = Run.get_context() 

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))


    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    #Save model for current iteration using C and max_iter values
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.C,args.max_iter))

if __name__ == '__main__':
    main()

    