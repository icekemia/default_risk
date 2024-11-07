
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreditCardBalanceAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, cc_balance):
        cc_balance_processed = cc_balance.copy()
        # 1. CREDIT_UTILIZATION_RATIO: Ratio of balance to credit limit
        cc_balance_processed['CREDIT_UTILIZATION_RATIO'] = cc_balance_processed['AMT_BALANCE'] / (cc_balance_processed['AMT_CREDIT_LIMIT_ACTUAL'] + 1)

        # 2. PAYMENT_BALANCE_RATIO: Ratio of payment to balance
        cc_balance_processed['PAYMENT_BALANCE_RATIO'] = cc_balance_processed['AMT_PAYMENT_CURRENT'] / (cc_balance_processed['AMT_BALANCE'] + 1)

        # 3. IS_OVERDUE: Flag for overdue payments
        cc_balance_processed['IS_OVERDUE'] = (cc_balance_processed['SK_DPD'] > 0).astype(int)

        # 4. TOTAL_DELINQUENT_MONTHS: Total months with overdue payments
        delinquency_count = cc_balance_processed[cc_balance_processed['SK_DPD'] > 0].groupby('SK_ID_CURR')['SK_DPD'].count().reset_index()
        delinquency_count.columns = ['SK_ID_CURR', 'TOTAL_DELINQUENT_MONTHS']
        cc_balance_processed = cc_balance_processed.merge(delinquency_count, on='SK_ID_CURR', how='left').fillna({'TOTAL_DELINQUENT_MONTHS': 0})

        # 5. ATM_DRAWINGS_RATIO: Ratio of ATM drawings to total drawings
        cc_balance_processed['ATM_DRAWINGS_RATIO'] = cc_balance_processed['CNT_DRAWINGS_ATM_CURRENT'] / (cc_balance_processed['AMT_DRAWINGS_CURRENT'] + 1)

        # 6. MAX_UTILIZATION: Maximum utilization ratio across months
        cc_balance_processed['MAX_UTILIZATION'] = cc_balance_processed.groupby('SK_ID_CURR')['CREDIT_UTILIZATION_RATIO'].transform('max')

        # Aggregations at SK_ID_CURR level
        credit_card_agg = cc_balance_processed.groupby('SK_ID_CURR').agg({
            'AMT_BALANCE': ['mean', 'max', 'min', 'sum'],                # Summary statistics of balance
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],                  # Summary statistics of credit limit
            'AMT_PAYMENT_CURRENT': ['sum', 'mean'],                      # Summary statistics of payments
            'SK_DPD': ['max', 'mean', 'sum'],                            # Summary statistics of overdue days
            'CREDIT_UTILIZATION_RATIO': ['mean', 'max'],                 # Average and max credit utilization
            'TOTAL_DELINQUENT_MONTHS': ['max'],                          # Total delinquent months per customer
            'ATM_DRAWINGS_RATIO': ['mean']                               # Average ATM drawings ratio
        }).reset_index()

        # Flatten column names and give meaningful names
        credit_card_agg.columns = ['_'.join(col).strip().upper() if col[1] else col[0].upper() for col in credit_card_agg.columns]
        credit_card_agg.rename(columns={
            'AMT_BALANCE_MEAN': 'AMT_BALANCE_MEAN_CC',
            'AMT_BALANCE_MAX': 'AMT_BALANCE_MAX_CC',
            'AMT_BALANCE_MIN': 'AMT_BALANCE_MIN_CC',
            'AMT_BALANCE_SUM': 'AMT_BALANCE_SUM_CC',
            'AMT_CREDIT_LIMIT_ACTUAL_MEAN': 'AMT_CREDIT_LIMIT_ACTUAL_MEAN_CC',
            'AMT_CREDIT_LIMIT_ACTUAL_MAX': 'AMT_CREDIT_LIMIT_ACTUAL_MAX_CC',
            'AMT_PAYMENT_CURRENT_SUM': 'AMT_PAYMENT_CURRENT_SUM_CC',
            'AMT_PAYMENT_CURRENT_MEAN': 'AMT_PAYMENT_CURRENT_MEAN_CC',
            'SK_DPD_MAX': 'SK_DPD_MAX_CC',
            'SK_DPD_MEAN': 'SK_DPD_MEAN_CC',
            'SK_DPD_SUM': 'SK_DPD_SUM_CC',
            'CREDIT_UTILIZATION_RATIO_MEAN': 'CREDIT_UTILIZATION_MEAN_CC',
            'CREDIT_UTILIZATION_RATIO_MAX': 'CREDIT_UTILIZATION_MAX_CC',
            'TOTAL_DELINQUENT_MONTHS_MAX': 'TOTAL_DELINQUENT_MONTHS_CC',
            'ATM_DRAWINGS_RATIO_MEAN': 'ATM_DRAWINGS_RATIO_MEAN_CC'
        }, inplace=True)

        return credit_card_agg
