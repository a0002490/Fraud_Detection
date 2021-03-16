# SCRIPT NAME:
# raw_data_simulation.py

# SCRIPT PURPOSE:
# simulate transaction, back office data and currency rate

# SCRIPT AUTHOR:
# Benhao Li

# INPUT DATASETS:
# 

# OUTPUT DATASET:
# transaction.csv
# back_office.csv
# currency.csv

# VERSION 1.0 2021/02/22

######################
#      IMPORTS       #
######################

import pandas as pd
import numpy as np
import os, sys, random, datetime
import itertools

import warnings
warnings.filterwarnings('ignore')


######################
#  DEFINE FUNCTIONS  #
######################
def main():
    # constants 
    output_transaction_file = './simulate_data/transaction.csv'
    output_back_office_file = './simulate_data/back_office.csv'
    output_currency_file = './simulate_data/currency.csv'
    n_trans = 100000

    # simulate currency rate
    currency_df = simulate_currency_rate()

    # simulate normal transactions
    trans_df = simulate_normal_transaction(currency_df, n_trans)

    # simulate unusual transactions
    trans_df, back_office_df = simulate_unusual_transaction(trans_df, n_trans)

    # save dataframes
    currency_df.to_csv(output_currency_file, index=False)
    trans_df.to_csv(output_transaction_file, index=False)
    back_office_df.to_csv(output_back_office_file, index=False)


def simulate_normal_transaction(currency_df, n_trans):

    # constants
    n_traders=100
    start_date=datetime.date(2000,1,1)
    end_date=datetime.date(2002,12,31)
    trans_start_hr=9
    trans_end_hr=17
    submit_start_hr=18
    submit_end_hr=20
    amend_type=["YES","NO"]
    amend_type_p=[0.1,0.9] 
    usual_currency_type=["AUD","USD","JPY","EUR","CHY","KRW","CAD"]
    usual_currency_type_p=[0.35,0.35,0.1,0.05,0.05,0.05,0.05]

    amount_start=10000
    amount_end=100000

    strike_ratio_low=0.5
    strike_ratio_high=1.5

    # set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Transaction ID 
    trans_df=pd.DataFrame(data={"Transaction_ID":list(map(lambda x:"transaction_"+"{:06}".format(x),
                                                 list(range(1,n_trans+1))))})

    # Trader ID - random trader
    trans_df['Trader_ID'] = list(map(lambda x:"trader_"+"{:03}".format(x),
                             np.random.choice(np.array(list(range(1,n_traders+1))),
                                              n_trans,replace=True)))
    # Transaction Date (Mon-Fri)
    workday_list = list(pd.bdate_range(start=start_date,end=end_date))
    trans_df["Transaction_Date"]=np.random.choice(np.array(workday_list),n_trans,replace=True)

    # Transcation time (9am - 5pm)
    trans_df['Transaction_Time'] = (trans_df['Transaction_Date']+\
                                    pd.to_timedelta(np.random.choice(np.array(range(trans_start_hr,trans_end_hr)),
                                                                     n_trans,replace=True),unit="h")+\
                                    pd.to_timedelta(np.random.choice(np.array(range(60)),n_trans,replace=True),unit="m")+\
                                    pd.to_timedelta(np.random.choice(np.array(range(60)),n_trans,replace=True),unit="S"))

    # Submit time
    trans_df['Submit_Time'] = (trans_df['Transaction_Date']+\
                                pd.to_timedelta(np.random.choice(np.array(range(submit_start_hr,submit_end_hr)),
                                                                 n_trans,replace=True),unit="h")+\
                                pd.to_timedelta(np.random.choice(np.array(range(60)),n_trans,replace=True),unit="m")+\
                                pd.to_timedelta(np.random.choice(np.array(range(60)),n_trans,replace=True),unit="S"))

    # Currency exchange type 1
    trans_df["Currency_Exchange_Type_From"]=np.random.choice(np.array(usual_currency_type),
                                                             n_trans,
                                                             replace=True,
                                                             p=np.array(usual_currency_type_p))

    # Currency exchange type 2
    trans_df['Currency_Exchange_Type_To'] = trans_df.apply(simulate_currency_exchange_type_to,
                                                            axis=1,
                                                            args=(usual_currency_type,usual_currency_type_p))

    # Amendment type
    trans_df["Amend_Type"]=np.random.choice(np.array(amend_type),
                                                     n_trans,
                                                     replace=True,
                                                     p=np.array(amend_type_p))


    # Quantity(Volume)
    amount_list = np.random.randint(low=amount_start, high=amount_end, size=(n_trans))
    trans_df["Volume"]=np.random.choice(np.array(amount_list),n_trans,replace=True)


    # Total Dollar Amount (Cost)
    currency_df['Date'] = pd.to_datetime(currency_df['Date'])
    trans_df = pd.merge(trans_df,
                        currency_df,
                        left_on=['Transaction_Date','Currency_Exchange_Type_From','Currency_Exchange_Type_To'],
                        right_on=['Date','Currency_From','Currency_To'],
                        how='left')

    trans_df['Unit_Cost'] = np.random.uniform(low=trans_df.Daily_Low,high= trans_df.Daily_High)
    trans_df["Total_Dollar_Amount_Cost"]=trans_df["Volume"]*trans_df["Unit_Cost"]

    # Total Dollar Amount (Strike Price)
    trans_df['Stike_Ratio'] = np.random.uniform(low=strike_ratio_low,high=strike_ratio_high, size=n_trans)
    trans_df['Unit_Sales'] = np.random.uniform(low=trans_df.Daily_Low,high=trans_df.Daily_High, size=n_trans) * trans_df['Stike_Ratio']
    trans_df["Total_Dollar_Amount_Strike"]=trans_df["Volume"]*trans_df["Unit_Sales"]

    # Total Profit
    trans_df['Total_Profit'] = trans_df["Total_Dollar_Amount_Strike"] - trans_df["Total_Dollar_Amount_Cost"]

    # clean up 
    trans_df.drop(['Date','Currency_From','Currency_To','Daily_High','Daily_Low','Stike_Ratio'], axis=1, inplace=True)

    return trans_df



def simulate_currency_exchange_type_to(row, usual_currency_type,usual_currency_type_p):
    curr_from = row['Currency_Exchange_Type_From'] # to drop currency type
    curr_from_id = usual_currency_type.index(curr_from) # find index of currency type in currency_from list
    if curr_from_id != len(usual_currency_type_p)-1:
        tmp_curr_type_list = usual_currency_type[0:curr_from_id]+usual_currency_type[curr_from_id+1:]
        tmp_curr_prob_list = usual_currency_type_p[0:curr_from_id]+usual_currency_type_p[curr_from_id+1:]
    else:
        tmp_curr_type_list = usual_currency_type[0:curr_from_id]
        tmp_curr_prob_list = usual_currency_type_p[0:curr_from_id]
        
    # rescale probability
    tmp_curr_prob_list = list(map(lambda x: x/(1-usual_currency_type_p[curr_from_id]), tmp_curr_prob_list))
    
    return np.random.choice(a=np.array(tmp_curr_type_list),
                            p=np.array(tmp_curr_prob_list))



def simulate_unusual_transaction(trans_df, n_trans):
    # set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # 1. update Currency_Exchange_Type_From from 5000 transactions
    n_select = 5000
    selected_trans_from = np.random.choice(np.array(trans_df['Transaction_ID']),n_select, replace=False)
    selected_trans_from_baht = np.random.choice(selected_trans_from,int(n_select*0.4), replace=False)
    selected_trans_from_mxn = np.setdiff1d(selected_trans_from,selected_trans_from_baht)

    trans_df['Currency_Exchange_Type_From'] = np.where(trans_df['Transaction_ID'].isin(list(selected_trans_from_baht)),
                                                      "BAHT", trans_df['Currency_Exchange_Type_From'])

    trans_df['Currency_Exchange_Type_From'] = np.where(trans_df['Transaction_ID'].isin(list(selected_trans_from_mxn)),
                                                      "MXN", trans_df['Currency_Exchange_Type_From'])

    # 2. update Currency_Exchange_Type_To from 5000 transactions
    n_select = 5000
    selected_trans_to = np.random.choice(np.setdiff1d(np.array(trans_df['Transaction_ID']),selected_trans_from),
                                         n_select, replace=False)
    selected_trans_to_baht = np.random.choice(selected_trans_to,int(n_select*0.4), replace=False)
    selected_trans_to_mxn = np.setdiff1d(selected_trans_to,selected_trans_to_baht)

    trans_df['Currency_Exchange_Type_To'] = np.where(trans_df['Transaction_ID'].isin(list(selected_trans_to_baht)),
                                                      "BAHT", trans_df['Currency_Exchange_Type_To'])

    trans_df['Currency_Exchange_Type_To'] = np.where(trans_df['Transaction_ID'].isin(list(selected_trans_to_mxn)),
                                                      "MXN", trans_df['Currency_Exchange_Type_To'])

    # save back office data
    back_office_df = trans_df.copy()

    # 3. update Unit_Cost from 3000 transactions
    n_select = 3000
    selected_trans_cost = np.random.choice(np.array(trans_df['Transaction_ID']),n_select, replace=False)

    def update_unit_cost_or_sales(row, selected_trans):
        if row['Transaction_ID'] in list(selected_trans):
            return np.random.choice(np.array(range(2,100)))
        else:
            return 1
        
    trans_df['unit_cost_update_ratio'] = trans_df.apply(update_unit_cost_or_sales, axis=1, args=(selected_trans_cost,))

    trans_df['Unit_Cost'] = trans_df['Unit_Cost'] * trans_df['unit_cost_update_ratio']
    trans_df['Total_Dollar_Amount_Cost'] = trans_df["Volume"]*trans_df["Unit_Cost"]
    trans_df['Total_Profit'] = trans_df["Total_Dollar_Amount_Strike"] - trans_df["Total_Dollar_Amount_Cost"]

    trans_df.drop(['unit_cost_update_ratio'], axis=1, inplace=True)

    # 4. update Unit_Sales from 3000 transactions
    n_select = 3000
    selected_trans_sales = np.random.choice(np.array(trans_df['Transaction_ID']), n_select, replace=False)
        
    trans_df['unit_sales_update_ratio'] = trans_df.apply(update_unit_cost_or_sales, axis=1, args=(selected_trans_sales,))

    trans_df['Unit_Sales'] = trans_df['Unit_Sales'] * trans_df['unit_sales_update_ratio']
    trans_df['Total_Dollar_Amount_Strike'] = trans_df["Volume"]*trans_df["Unit_Sales"]
    trans_df['Total_Profit'] = trans_df["Total_Dollar_Amount_Strike"] - trans_df["Total_Dollar_Amount_Cost"]

    trans_df.drop(['unit_sales_update_ratio'], axis=1, inplace=True)

    # 5. update Total_Profit from 1000 transactions (*100)
    n_select = 1000
    selected_trans_profit = np.random.choice(np.array(trans_df['Transaction_ID']), n_select, replace=False)
    trans_df['Total_Profit'] = np.where(trans_df['Transaction_ID'].isin(list(selected_trans_profit)),
                                       trans_df['Total_Profit']*100, trans_df['Total_Profit'])

    # 6. add 500 transactions (same content but different transaction ID)
    n_select = 500
    selected_trans_diff_ID = np.random.choice(np.array(trans_df['Transaction_ID']), n_select, replace=False)
    tmp_df = trans_df.loc[trans_df['Transaction_ID'].isin(list(selected_trans_diff_ID))]
    tmp_df = tmp_df.assign(Transaction_ID = list(map(lambda x:"transaction_"+"{:06}".format(x),
                                                 list(range(n_trans+1,n_trans+n_select+1)))))

    trans_df = pd.concat([trans_df, tmp_df], ignore_index=True)

    return trans_df, back_office_df


def simulate_currency_rate():

    # constants
    start_date=datetime.date(2000,1,1)
    end_date=datetime.date(2002,12,31)

    usual_currency_type=["AUD","USD","JPY","EUR","CHY","KRW","CAD"]

    usd_to_currency_dict={
        "AUD":[1.5,2],
        "JPY":[100,130],
        "EUR":[0.9,1.2],
        "CHY":[7,8],
        "KRW":[900,1200],
        "CAD":[1.1,1.4]
    }

    # set seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    curr_ratio_df = calculate_currency_ratio(usual_currency_type, usd_to_currency_dict)
    currency_date=list(pd.bdate_range(start=start_date,end=end_date))
    df = pd.DataFrame()

    curr_perm = list(itertools.permutations(usual_currency_type, 2))
    for (curr1, curr2) in curr_perm:
        tmp_df = pd.DataFrame()
        tmp_df["Date"]=currency_date
        tmp_df["Currency_From"]=curr1
        tmp_df["Currency_To"]=curr2
        df = pd.concat([df, tmp_df], ignore_index=True)

    # get daily ratio range
    df = pd.merge(df, curr_ratio_df, 
                 on=['Currency_From','Currency_To'],
                 how='left')

    df["Value1"]= np.random.uniform(low=df["Ratio_min"], high=df['Ratio_max'])
    df["Value2"]= np.random.uniform(low=df["Ratio_min"], high=df['Ratio_max'])

    df["Daily_High"]= df[["Value1", "Value2"]].max(axis=1)
    df["Daily_Low"]= df[["Value1", "Value2"]].min(axis=1)

    # drop Ratio_min, Ratio_max, Value1, Value2
    df.drop(['Ratio_min', 'Ratio_max', 'Value1', 'Value2'],axis=1, inplace=True)

    return df


# input: usual_currency_type
# output: dataframe - curr1, curr2, ratio min, ratio max
def calculate_currency_ratio(usual_currency_type, usd_to_currency_dict):
    curr_perm = list(itertools.permutations(usual_currency_type, 2))
    df = pd.DataFrame()
    # from curr1 to curr2
    for (curr1, curr2) in curr_perm: 
        
        if curr1 == 'USD':
            ratio_min = 1
            ratio_max = 1
            
        else:
            ratio_min = 1/usd_to_currency_dict[curr1][1]
            ratio_max = 1/usd_to_currency_dict[curr1][0]
            
        tmp_df = pd.DataFrame(data={'Currency_From':[curr1],
                                    'Currency_To':[curr2],
                                    'Ratio_min':[ratio_min],
                                    'Ratio_max':[ratio_max]})
        df = pd.concat([df, tmp_df], ignore_index=True)
        
    return df


######################
# CALL MAIN FUNCTION #
######################

if __name__ == '__main__':
    main()
