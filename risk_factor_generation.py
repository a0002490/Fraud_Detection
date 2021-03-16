# SCRIPT NAME:
# risk_factor_generation.py

# SCRIPT PURPOSE:
# determine if each rule is violated at transaction level

# SCRIPT AUTHOR:
# Benhao Li

# INPUT DATASETS:
# transaction.csv
# back_office.csv

# OUTPUT DATASET:
# rule_violations.csv

# VERSION 1.0 2021/03/07

######################
#      IMPORTS       #
######################

import pandas as pd
import numpy as np
import os, sys, random, datetime

import warnings
warnings.filterwarnings('ignore')


######################
#  DEFINE FUNCTIONS  #
######################
def main():

    # constants 
    transaction_file = './simulate_data/transaction.csv'
    back_office_file = './simulate_data/back_office.csv'
    output_file = './simulate_data/rule_violations.csv'

    rare_currency = ['BAHT','MXN']
    soft_limit = 1000000 # 1M
    hard_limit = 2000000 # 2M

    # read in data and format columns
    trans_df = pd.read_csv(transaction_file)
    back_office_df = pd.read_csv(back_office_file)

    trans_df['Transaction_Date'] = pd.to_datetime(trans_df['Transaction_Date'])

    # initiate dataframe to save risk factors
    df = pd.DataFrame(data={'Transaction_ID':trans_df['Transaction_ID'].tolist()})

    # 1. From currency is rare or not rare
    rule_df = currency_from_is_rare(trans_df, rare_currency)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 2. To currency is rare or not rare
    rule_df = currency_to_is_rare(trans_df, rare_currency)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 3. Total Cost is unusual high - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_total_cost(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 4. Total strike price is unusual high - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_total_strike(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 5. Single transaction amount is unusual high - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_volume(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 6. Single transaction amount is unusual low - Z-SCORE < threshold (default threshold: -3)
    rule_df = unusual_low_volume(trans_df, -3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 7. Employee submit multiple transactions with same content but different transaction IDs
    rule_df = multi_transaction_same_content(trans_df)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 8. Mismatched submitted data with back office data
    rule_df = mismatch_with_back_office_data(trans_df, back_office_df)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 9. Amended the transaction
    rule_df = amended_transaction(trans_df)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 10. unusual high transaction counts per day - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_daily_transaction_cnt(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 11. unusual high transaction counts per week - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_weekly_transaction_cnt(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 12. unusual high transaction counts per month - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_monthly_transaction_cnt(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 13. unusual high month over month change rate - Z-SCORE > threshold (default threshold: 3)
    rule_df = unusual_high_mom_transaction_cnt_change_rate(trans_df, 3)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 14. break soft limit (threshold) per month (on cost)
    rule_df = break_soft_limit(trans_df, soft_limit)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # 15. break hard limit per month (on cost)
    rule_df = break_hard_limie(trans_df, hard_limit)
    df = pd.merge(df, rule_df, on='Transaction_ID', how='left')

    # save file
    df.to_csv(output_file, index=False)



# From currency is rare or not rare
def currency_from_is_rare(trans_df, rare_currency):

    trans_df['RULE_CURRENCY_FROM_IS_RARE'] = np.where(trans_df['Currency_Exchange_Type_From'].isin(rare_currency),
                                                      1, 0)

    res = trans_df[['Transaction_ID', 'RULE_CURRENCY_FROM_IS_RARE']].drop_duplicates()

    trans_df.drop(['RULE_CURRENCY_FROM_IS_RARE'], axis=1, inplace=True)

    return res


# To currency is rare or not rare
def currency_to_is_rare(trans_df, rare_currency):
    trans_df['RULE_CURRENCY_TO_IS_RARE'] = np.where(trans_df['Currency_Exchange_Type_To'].isin(rare_currency),
                                                    1, 0)

    res = trans_df[['Transaction_ID', 'RULE_CURRENCY_TO_IS_RARE']].drop_duplicates()

    trans_df.drop(['RULE_CURRENCY_TO_IS_RARE'], axis=1, inplace=True)

    return res



# Total Cost is unusual high
def unusual_high_total_cost(trans_df, threshold):
    trans_df = trans_df.assign(TOTAL_COST_MEAN = trans_df.groupby(['Currency_Exchange_Type_From',
                                                                   'Currency_Exchange_Type_To'])['Total_Dollar_Amount_Cost']
                                                         .transform('mean'),
                               TOTAL_COST_STD = trans_df.groupby(['Currency_Exchange_Type_From',
                                                                   'Currency_Exchange_Type_To'])['Total_Dollar_Amount_Cost']
                                                         .transform('std'))
    trans_df['TOTAL_COST_ZSCORE']=(trans_df['Total_Dollar_Amount_Cost']-trans_df['TOTAL_COST_MEAN'])/trans_df['TOTAL_COST_STD']
    trans_df['RULE_UNUSUAL_HIGH_TOTAL_COST'] = np.where(trans_df['TOTAL_COST_ZSCORE']>threshold,
                                                        1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_TOTAL_COST']].drop_duplicates()

    trans_df.drop(['TOTAL_COST_MEAN',
                   'TOTAL_COST_STD',
                   'TOTAL_COST_ZSCORE',
                   'RULE_UNUSUAL_HIGH_TOTAL_COST'], axis=1, inplace=True)

    return res


# Total strike price is unusual high
def unusual_high_total_strike(trans_df, threshold):
    trans_df = trans_df.assign(TOTAL_STRIKE_MEAN = trans_df.groupby(['Currency_Exchange_Type_From',
                                                                     'Currency_Exchange_Type_To'])['Total_Dollar_Amount_Strike']
                                                           .transform('mean'),
                               TOTAL_STRIKE_STD = trans_df.groupby(['Currency_Exchange_Type_From',
                                                                    'Currency_Exchange_Type_To'])['Total_Dollar_Amount_Strike']
                                                          .transform('std'))
    trans_df['TOTAL_STRIKE_ZSCORE']=(trans_df['Total_Dollar_Amount_Strike']-trans_df['TOTAL_STRIKE_MEAN'])/trans_df['TOTAL_STRIKE_STD']
    trans_df['RULE_UNUSUAL_HIGH_TOTAL_STRIKE'] = np.where(trans_df['TOTAL_STRIKE_ZSCORE']>threshold,
                                                          1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_TOTAL_STRIKE']].drop_duplicates()

    trans_df.drop(['TOTAL_STRIKE_MEAN',
                   'TOTAL_STRIKE_STD',
                   'TOTAL_STRIKE_ZSCORE',
                   'RULE_UNUSUAL_HIGH_TOTAL_STRIKE'], axis=1, inplace=True)

    return res


# Single transaction amount is unusual high
def unusual_high_volume(trans_df, threshold):
    trans_df = trans_df.assign(VOL_MEAN = trans_df.groupby(['Currency_Exchange_Type_From',
                                                            'Currency_Exchange_Type_To'])['Volume']
                                                  .transform('mean'),
                               VOL_STD = trans_df.groupby(['Currency_Exchange_Type_From',
                                                           'Currency_Exchange_Type_To'])['Volume']
                                                 .transform('std'))

    trans_df['VOL_ZSCORE']=(trans_df['Volume']-trans_df['VOL_MEAN'])/trans_df['VOL_STD']
    trans_df['RULE_UNUSUAL_HIGH_VOLUME'] = np.where(trans_df['VOL_ZSCORE']>threshold, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_VOLUME']].drop_duplicates()

    trans_df.drop(['VOL_MEAN',
                   'VOL_STD',
                   'VOL_ZSCORE',
                   'RULE_UNUSUAL_HIGH_VOLUME'], axis=1, inplace=True)

    return res


# Single transaction amount is unusual low
def unusual_low_volume(trans_df, threshold):
    trans_df = trans_df.assign(VOL_MEAN = trans_df.groupby(['Currency_Exchange_Type_From',
                                                            'Currency_Exchange_Type_To'])['Volume']
                                                  .transform('mean'),
                               VOL_STD = trans_df.groupby(['Currency_Exchange_Type_From',
                                                           'Currency_Exchange_Type_To'])['Volume']
                                                 .transform('std'))

    trans_df['VOL_ZSCORE']=(trans_df['Volume']-trans_df['VOL_MEAN'])/trans_df['VOL_STD']
    trans_df['RULE_UNUSUAL_LOW_VOLUME'] = np.where(trans_df['VOL_ZSCORE']<threshold, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_LOW_VOLUME']].drop_duplicates()

    trans_df.drop(['VOL_MEAN',
                   'VOL_STD',
                   'VOL_ZSCORE',
                   'RULE_UNUSUAL_LOW_VOLUME'], axis=1, inplace=True)

    return res


# Employee submit multiple transactions with same content but different transaction IDs
def multi_transaction_same_content(trans_df):
    selected_cols = list(filter(lambda x: x != "Transaction_ID", trans_df.columns))
    trans_df = trans_df.assign(CNT = trans_df.groupby(selected_cols)['Transaction_ID']
                                             .transform('nunique'))

    trans_df['RULE_MULTI_TRANS_SAME_CONTENT'] = np.where(trans_df['CNT']>1, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_MULTI_TRANS_SAME_CONTENT']].drop_duplicates()

    trans_df.drop(['CNT',
                   'RULE_MULTI_TRANS_SAME_CONTENT'], axis=1, inplace=True)

    return res


# Mismatched submitted data with back office data
def mismatch_with_back_office_data(trans_df, back_office_df):
    back_office_df['Transaction_Date'] = pd.to_datetime(back_office_df['Transaction_Date'])

    selected_cols = list(filter(lambda x: x != "Transaction_ID", trans_df.columns))
    back_office_df.columns=['Transaction_ID'] + list(map(lambda x: 'BACK_OFFICE_'+x, selected_cols))

    trans_df['RULE_MISMATCH_BACK_OFFICE_DATA'] = 0
    for col in selected_cols:
        tmp_df = pd.merge(trans_df[['Transaction_ID',col]],
                         back_office_df[['Transaction_ID','BACK_OFFICE_'+col]],
                         on='Transaction_ID',
                         how='left')
        mismatched_trans = tmp_df.loc[tmp_df[col] != tmp_df['BACK_OFFICE_'+col],
                                     'Transaction_ID'].tolist()

        trans_df['RULE_MISMATCH_BACK_OFFICE_DATA'] = np.where(trans_df['Transaction_ID'].isin(mismatched_trans),
                                                             1, trans_df['RULE_MISMATCH_BACK_OFFICE_DATA'])

    res = trans_df[['Transaction_ID', 'RULE_MISMATCH_BACK_OFFICE_DATA']].drop_duplicates()

    trans_df.drop(['RULE_MISMATCH_BACK_OFFICE_DATA'], axis=1, inplace=True)

    return res


# Amended the transaction
def amended_transaction(trans_df):
    trans_df['RULE_AMENDED_TRANSACTION'] = np.where(trans_df['Amend_Type']=='YES',
                                                    1, 0)

    res = trans_df[['Transaction_ID', 'RULE_AMENDED_TRANSACTION']].drop_duplicates()

    trans_df.drop(['RULE_AMENDED_TRANSACTION'], axis=1, inplace=True)

    return res


# unusual high transaction counts per day
def unusual_high_daily_transaction_cnt(trans_df, threshold):
    trans_df = trans_df.assign(CNT = trans_df.groupby(['Trader_ID','Transaction_Date'])['Transaction_ID']
                                             .transform('nunique'))
    CNT_MEAN = trans_df['CNT'].mean()
    CNT_SD = trans_df['CNT'].std()

    trans_df['CNT_ZSCORE']=(trans_df['CNT']-CNT_MEAN)/CNT_SD

    trans_df['RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_DAY'] = np.where(trans_df['CNT_ZSCORE']>threshold, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_DAY']].drop_duplicates()

    trans_df.drop(['CNT',
                   'CNT_ZSCORE',
                   'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_DAY'], axis=1, inplace=True)

    return res


# unusual high transaction counts per week
def unusual_high_weekly_transaction_cnt(trans_df, threshold):
    trans_df['YEAR'] = trans_df['Transaction_Date'].dt.year
    trans_df['WEEK'] = trans_df['Transaction_Date'].dt.week

    trans_df = trans_df.assign(CNT = trans_df.groupby(['Trader_ID','YEAR','WEEK'])['Transaction_ID']
                                             .transform('nunique'))
    CNT_MEAN = trans_df['CNT'].mean()
    CNT_SD = trans_df['CNT'].std()

    trans_df['CNT_ZSCORE']=(trans_df['CNT']-CNT_MEAN)/CNT_SD

    trans_df['RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_WEEK'] = np.where(trans_df['CNT_ZSCORE']>threshold, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_WEEK']].drop_duplicates()

    trans_df.drop(['CNT',
                   'CNT_ZSCORE',
                   'YEAR',
                   'WEEK',
                   'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_WEEK'], axis=1, inplace=True)
    return res


# unusual high transaction counts per month
def unusual_high_monthly_transaction_cnt(trans_df, threshold):
    trans_df['YEAR'] = trans_df['Transaction_Date'].dt.year
    trans_df['MONTH'] = trans_df['Transaction_Date'].dt.month

    trans_df = trans_df.assign(CNT = trans_df.groupby(['Trader_ID','YEAR','MONTH'])['Transaction_ID']
                                             .transform('nunique'))
    CNT_MEAN = trans_df['CNT'].mean()
    CNT_SD = trans_df['CNT'].std()

    trans_df['CNT_ZSCORE']=(trans_df['CNT']-CNT_MEAN)/CNT_SD

    trans_df['RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_MONTH'] = np.where(trans_df['CNT_ZSCORE']>threshold, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_MONTH']].drop_duplicates()

    trans_df.drop(['CNT',
                   'CNT_ZSCORE',
                   'YEAR',
                   'MONTH',
                   'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_PER_MONTH'], axis=1, inplace=True)

    return res


# unusual high month over month change rate
def unusual_high_mom_transaction_cnt_change_rate(trans_df, threshold):
    trans_df['YEAR'] = trans_df['Transaction_Date'].dt.year
    trans_df['MONTH'] = trans_df['Transaction_Date'].dt.month
    trans_df['PSEUDO_DATE'] = pd.to_datetime(dict(year=trans_df.YEAR, 
                                                  month=trans_df.MONTH, 
                                                  day=1))

    trans_df = trans_df.assign(CNT = trans_df.groupby(['Trader_ID','PSEUDO_DATE'])['Transaction_ID']
                                             .transform('nunique'))

    # init another dataframe loop through all possible months
    start_date = trans_df['PSEUDO_DATE'].min()
    end_date = trans_df['PSEUDO_DATE'].max()
    date_list = pd.date_range(start=start_date, end=end_date, freq='MS')
    trader_list = sorted(list(set(trans_df['Trader_ID'])))

    idx = pd.MultiIndex.from_product([trader_list, date_list], names = ["Trader_ID", "PSEUDO_DATE"])
    df = pd.DataFrame(index = idx).reset_index()

    # merge two dataframes to get number of transactions per trader per month, fill NA as 0
    df = pd.merge(df, 
                  trans_df[['Trader_ID', 'PSEUDO_DATE', 'CNT']].drop_duplicates(),
                  on=["Trader_ID", "PSEUDO_DATE"],
                  how='left')
    df['CNT'] = df['CNT'].fillna(0)
    df.sort_values(['Trader_ID', 'PSEUDO_DATE'], inplace=True)

    # calculate month over month change rate, and merge back to transaction level
    df = df.assign(PREV_CNT = df.groupby(['Trader_ID'])['CNT'].shift(1))
    df['MOM_CHANGE_RATE'] = (df['CNT'] - df['PREV_CNT'])/df['PREV_CNT']
    trans_df = pd.merge(trans_df, 
                        df[['Trader_ID', 'PSEUDO_DATE', 'MOM_CHANGE_RATE']],
                        on=['Trader_ID', 'PSEUDO_DATE'],
                        how='left')

    # find outliers
    CHANGE_RATE_MEAN = trans_df['MOM_CHANGE_RATE'].mean()
    CHANGE_RATE_SD = trans_df['MOM_CHANGE_RATE'].std()

    trans_df['ZSCORE']=(trans_df['MOM_CHANGE_RATE']-CHANGE_RATE_MEAN)/CHANGE_RATE_SD

    trans_df['RULE_UNUSUAL_HIGH_TRANSACTION_CNT_MOM_CHANGE_RATE'] = np.where(trans_df['ZSCORE']>threshold, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_MOM_CHANGE_RATE']].drop_duplicates()

    trans_df.drop(['CNT',
                   'MOM_CHANGE_RATE',
                   'ZSCORE',
                   'YEAR',
                   'MONTH',
                   'WEEK',
                   'PSEUDO_DATE',
                   'RULE_UNUSUAL_HIGH_TRANSACTION_CNT_MOM_CHANGE_RATE'], axis=1, inplace=True)

    return res


# break soft limit (threshold) per month (on cost)
def break_soft_limit(trans_df, soft_limit):
    trans_df['RULE_BREAK_SOFT_LIMIT'] = np.where(trans_df['Total_Dollar_Amount_Cost']>soft_limit, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_BREAK_SOFT_LIMIT']].drop_duplicates()

    trans_df.drop(['RULE_BREAK_SOFT_LIMIT'], axis=1, inplace=True)

    return res


# break hard limit per month (on cost)
def break_hard_limie(trans_df, hard_limit):

    trans_df['RULE_BREAK_HARD_LIMIT'] = np.where(trans_df['Total_Dollar_Amount_Cost']>hard_limit, 1, 0)

    res = trans_df[['Transaction_ID', 'RULE_BREAK_HARD_LIMIT']].drop_duplicates()

    trans_df.drop(['RULE_BREAK_HARD_LIMIT'], axis=1, inplace=True)

    return res


######################
# CALL MAIN FUNCTION #
######################

if __name__ == '__main__':
    main()
