# -*- coding: utf-8 -*-
"""
Created in May 2022

@author: Furiend DA

Last modified on July 9th 2022 by CaiCai
"""
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm


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


class CBRecommend():

    def __init__(self, df_content_recommend):
        self.df_content_recommend = df_content_recommend

    def cosine_sim(self, v1, v2):
        """
        This function calculates the cosine similarity between two vectors
        
        Args:
            v1 (list) : the input vector 1 to be compared similarity
            v2 (list) : the input vector 2 to be compared similarity
        
        Returns:
            cos_sim (float) : the dataframe with specified columns replaced by one-hot encoding results
        """
        cos_sim = sum(dot(v1, v2) / (norm(v1) * norm(v2)))

        return cos_sim

    def recommend(self, pet_id, num_rec):
        """
        This function calculate similarity of input pet_id vector w.r.t all other vectors
        and returns top n user specified pets.
        
        Args:
            self (dataframe) : the dataframe to work on
            pet_id (string) : the input pet_id to find most similar content-based vectors
            num_rec (int) : the number of recommendation
        
        Returns:
            top_num_rec (dataframe) : the num_rec of vectors that are most similar to input pet_id
        """

        # calculate similarity of input pet_id vector w.r.t all other vectors
        input_vec = self.df_content_recommend.loc[pet_id].values
        self.df_content_recommend['sim'] = self.df_content_recommend.apply(
            lambda x: self.cosine_sim(input_vec, x.values), axis=1)

        # returns top n user specified books
        top_num_rec = self.df_content_recommend.nlargest(columns='sim',
                                                         n=num_rec)

        return top_num_rec


if __name__ == '__main__':
    # set working directory
    work_dir = "D:/0_Recommendation/furiends-ai-animal-adoption-system"
    os.chdir(work_dir)
    print("Current working directory:", os.getcwd())

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
                            col_to_be_ohe='Breed1')  # whether OHE
    pet_df = one_hot_encode(df_to_be_ohe=pet_df,
                            col_to_be_ohe='Breed2')  # whether OHE
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Color1')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Color2')
    pet_df = one_hot_encode(df_to_be_ohe=pet_df, col_to_be_ohe='Color3')

    # drop redundant columns
    cols = [
        'Name', 'Fee', 'State', 'RescuerID', 'VideoAmt', 'PhotoAmt',
        'Description', 'AdoptionSpeed', 'Quantity'
    ]
    pet_df.drop(columns=cols, inplace=True)
    pet_df.set_index('PetID', inplace=True)

    # set the pet_id as the target content
    pet_id_target = pet_df.index[:1]
    # set the number of output recommendation outputs
    pet_num_rec = 5
    # ran on a sample as an example
    content_recomend_system = CBRecommend(df_content_recommend=pet_df)
    print(
        content_recomend_system.recommend(pet_id=pet_id_target,
                                          num_rec=pet_num_rec))
