# SCRIPT NAME:
# ui_preparation.py

# SCRIPT PURPOSE:
# prepare data for the dashboard

# SCRIPT AUTHOR:
# Benhao Li

# INPUT DATASETS:
# transaction.csv
# rule_violations.csv
# transaction_score.csv

# OUTPUT DATASET:
# ui_input.csv

# VERSION 1.0 2021/03/13

######################
#      IMPORTS       #
######################

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


######################
#  DEFINE FUNCTIONS  #
######################
def main():

    # constants 
    transaction_file = './simulate_data/transaction.csv'
    transaction_score_file = './simulate_data/transaction_score.csv'
    output_file = './simulate_data/ui_input.csv'

    # read in data and format columns
    trans_df = pd.read_csv(transaction_file)
    transaction_score_df = pd.read_csv(transaction_score_file)

    # select Transaction_ID, Trader_ID, Transaction_Date, Total_Dollar_Amount_Cost in transaction table
    # select Rule, Transaction_Score from transaction score table
    trans_df['Transaction_Date'] = pd.to_datetime(trans_df['Transaction_Date']).dt.date
    df = pd.merge(trans_df[['Transaction_ID',
                            'Trader_ID',
                            'Transaction_Date',
                            'Total_Dollar_Amount_Cost']].drop_duplicates(),
                    transaction_score_df[['Transaction_ID',
                                          'Rule',
                                          'Transaction_Score']].drop_duplicates(),
                    on='Transaction_ID',
                    how='left')

    # fill missing transaction score
    df['Transaction_Score'] = df['Transaction_Score'].fillna(value=0)

    # save output
    df.to_csv(output_file, index=False)



######################
# CALL MAIN FUNCTION #
######################

if __name__ == '__main__':
    main()
