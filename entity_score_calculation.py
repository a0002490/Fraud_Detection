# SCRIPT NAME:
# entity_score_calculation.py

# SCRIPT PURPOSE:
# calculate transaction score

# SCRIPT AUTHOR:
# Benhao Li

# INPUT DATASETS:
# rule_violations.csv
# rule_weight.csv

# OUTPUT DATASET:
# transaction_score.csv

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
    risk_factor_file = './simulate_data/rule_violations.csv'
    rule_weight_file = './../rule_weight.csv'
    output_file = './simulate_data/transaction_score.csv'

    # read in data 
    rule_df = pd.read_csv(risk_factor_file) 
    weight_df = pd.read_csv(rule_weight_file)

    # reshape rule_df from wide to long format
    rule_df_long = pd.melt(rule_df, 
                           id_vars='Transaction_ID', 
                           value_vars=list(filter(lambda x: x != "Transaction_ID", 
                                                  rule_df.columns.tolist())),
                           var_name = 'RISK_FACTOR',
                           value_name = 'VIOLATION')

    # filter only violated rules
    rule_df_long = rule_df_long.loc[rule_df_long['VIOLATION'] == 1]

    # get rule weight
    df = pd.merge(rule_df_long, weight_df, on='RISK_FACTOR', how='left')
    max_weight = weight_df['Weight'].sum()
    
    # aggregate to transation level
    df = df.assign(Transaction_Score = df.groupby(['Transaction_ID'])['Weight'].transform('sum'))
    
    # rescale transaction score from 0 to 100
    df['Transaction_Score'] = df['Transaction_Score']/max_weight * 100

    # save output
    df.to_csv(output_file, index=False)



######################
# CALL MAIN FUNCTION #
######################

if __name__ == '__main__':
    main()
