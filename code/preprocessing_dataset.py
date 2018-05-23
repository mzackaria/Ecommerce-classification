# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:30:57 2018

@author: zakaria messai
"""


def preprocess_dataset(df):
    
    import numpy as np
    import pandas as pd
    import datetime as dt
    
    df = df[~df[['CustomerID']].isnull().values]
    df = df[df[['UnitPrice']].values > 0]
    
    df[['Country']] = df[['Country']].fillna('')
    df[['Country']] = np.array(df[['Country']].values == 'United Kingdom').astype(int)
    df.rename(columns={'Country': 'isUnitedKingdom'}, inplace=True)
    
    df = df.drop_duplicates()
    
    df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
    df['Cancelled'] = df['Quantity'] < 0
    
    NOW = dt.datetime(2011, 12, 10)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    #recency: days since last purchase
    recency = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days})
    recency['InvoiceDate'] = recency['InvoiceDate'].astype(int)
    recency.rename(columns={'InvoiceDate':'recency'}, inplace=True)
    
    #frequency: number of times the user bought something on the site
    frequency = df.groupby('CustomerID').agg({'InvoiceNo': lambda x: len(x)})
    frequency.rename(columns={'InvoiceNo': 'frequency'}, inplace=True)
    
    #monetary value spent on the site
    monetary_value = df.groupby('CustomerID').agg({'TotalPrice': lambda x: x.sum()})
    monetary_value.rename(columns={'TotalPrice': 'monetary_value'}, inplace=True)
    
    #mean of total price
    mean_monetary_value = df.groupby('CustomerID').agg({'TotalPrice': lambda x: x.sum()/len(x)})
    mean_monetary_value.rename(columns={'TotalPrice': 'mean_monetary_value'}, inplace=True)
    
    #mean period between two purchases
    mean_period_bt_2 = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (x.max() - x.min()).days/len(x)})
    mean_period_bt_2.rename(columns={'InvoiceDate': 'mean_period_bt_2'}, inplace=True)
    
    #period in days since the first purchase
    period_on_the_site = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.min()).days})
    period_on_the_site.rename(columns={'InvoiceDate': 'period_on_the_site'}, inplace=True)
    
    #monetary value on the site by day
    a = monetary_value.values
    b = period_on_the_site.values
    monetary_value_onsite = pd.DataFrame(np.divide(a, b, out=np.zeros_like(a), where=b != 0), index=monetary_value.index, columns=['monetary_value_on_site'])
    
    #isUnitedKingdom: is user from united kingdom
    is_united_kingdom = df.groupby('CustomerID').agg({'isUnitedKingdom': lambda x: sum(x)/len(x)})
    
    #frequency: number of times the user bought something on the site
    mean_quantity = df.groupby('CustomerID').agg({'Quantity': lambda x: x.mean()})
    mean_quantity.rename(columns={'Quantity': 'mean_quantity'}, inplace=True)
    
    #frequency: number of times the user bought something on the site
    mean_unit_price = df.groupby('CustomerID').agg({'UnitPrice': lambda x: x.mean()})
    mean_unit_price.rename(columns={'UnitPrice': 'mean_unit_price'}, inplace=True)
    
    return pd.concat([monetary_value, 
                      mean_monetary_value, 
                      mean_period_bt_2, 
                      monetary_value_onsite, 
                      mean_quantity, 
                      mean_unit_price, 
                      is_united_kingdom], axis=1)
