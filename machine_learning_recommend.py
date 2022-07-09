# -*- coding: utf-8 -*-
"""
Created in June 2022

@author: Furiend DA

Last modified on July 9th 2022 by CaiCai
"""
import warnings

warnings.filterwarnings("ignore")
import os
import logging
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV


def normalize(data_to_be_normalized):
    """
    This function normalizes the input data to be between 0 and 1
    
    Args:
        data_to_be_normalized (list): the data to be normalized
    
    Returns:
        data_normalized (list): the data already normalized
    """
    min_val = min(data_to_be_normalized)
    if min_val < 0:
        data_to_be_normalized = [
            x + abs(min_val) for x in data_to_be_normalized
        ]
    max_val = max(data_to_be_normalized)
    data_normalized = [x / max_val for x in data_to_be_normalized]

    return data_normalized


def one_hot_encode(df_to_be_ohe, col_to_be_ohe):
    """
    This function does one-hot encoding on the specified column of input dataframe and 
    replace the specificed columns with the one-hot encoding results of the input dataframe
    
    Args:
        df_to_be_ohe (dataframe) : the dataframe to edit
        col_to_be_ohe (string) : the column to be one-hot encoding
    
    Returns:
        df_ohe : the dataframe with specified columns replaced by one-hot encoding results
    """

    df_ohe = pd.get_dummies(df_to_be_ohe[col_to_be_ohe])
    df_ohe.reset_index(drop=True, inplace=True)
    df_ohe = pd.concat([df_to_be_ohe, df_ohe], axis=1)
    df_ohe = df_ohe.drop(columns=col_to_be_ohe)

    return df_ohe


def random_forest_classifer(input_df, cross_validation_flag=False):
    """
    This function does random forest classifier to predict the adoption speed,
    which is regarded as the likeness by users in our context.
    
    Args:
        input_df (dataframe) : the dataframe to work on
        cross_validation_flag (boolean) : the flag indicating whether to do cross validation 
    
    Returns:
        rf_classifer : the applied random forest classifier
    """
    # prepare data
    X = input_df.copy()
    X.drop(columns='AdoptionSpeed', inplace=True)  # features
    y = input_df['AdoptionSpeed'].copy()  # labels

    # split dataset into training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)  # 70% training and 30% test

    # do cross validation if specified
    if cross_validation_flag == True:
        n_estimators = [
            int(x) for x in np.linspace(start=200, stop=2000, num=10)
        ]  # number of trees in random forest
        max_features = ['auto', 'sqrt']  # number of features at every split
        max_depth = [int(x)
                     for x in np.linspace(100, 500, num=11)]  # max depth
        max_depth.append(None)

        # create random grid
        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth
        }

        # random search of parameters
        rfc_random = RandomizedSearchCV(estimator=rfc,
                                        param_distributions=random_grid,
                                        n_iter=100,
                                        cv=3,
                                        verbose=2,
                                        random_state=42,
                                        n_jobs=-1)

        # fit the model
        rfc_random.fit(X_train, y_train)

        # print results
        print(rfc_random.best_params_)

    # create a classifier
    rf_classifer = RandomForestClassifier(n_estimators=1400,
                                          max_features='auto',
                                          max_depth=220)

    # train the classifier
    rf_classifer.fit(X_train, y_train)

    # predict the test dataset
    y_pred = rf_classifer.predict(X_test)

    # print accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # print feature importances
    feature_imp = pd.Series(rf_classifer.feature_importances_,
                            index=X.columns).sort_values(ascending=False)
    print('Feature Importance:', feature_imp)

    return rf_classifer


if __name__ == '__main__':
    # set working directory
    work_dir = "D:/0_Recommendation/furiends-ai-animal-adoption-system"
    os.chdir(work_dir)
    logging.info("Current working directory:", os.getcwd())

    # set data path
    data_path = './data/pet_data.csv'
    # import data
    pet_df = pd.read_csv(data_path)

    # normalize numerical attributes
    pet_df['Age'] = normalize(pet_df['Age'].values)

    # one-hot encode categorical attributes
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Type')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Gender')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='MaturitySize')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='FurLength')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Vaccinated')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Dewormed')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Sterilized')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Health')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df,
                            col_to_be_ohe='Breed1')  # whether one_hot_encode
    pet_df = one_hot_encode(df_to_be_ohe=pet_df,
                            col_to_be_ohe='Breed2')  # whether one_hot_encode
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Color1')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Color2')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Color3')

    # drop redundant columns
    cols1 = [
        'Name', 'Fee', 'State', 'RescuerID', 'VideoAmt', 'PhotoAmt',
        'Description', 'Quantity'
    ]
    pet_df.drop(columns=cols1, inplace=True)
    pet_df.set_index('PetID', inplace=True)

    # run random forest classifier
    rf_classifer = random_forest_classifer(pet_df, cross_validation_flag=False)
