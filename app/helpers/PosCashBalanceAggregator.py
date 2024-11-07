
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PosCashBalanceAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, pos_cash):
        # Create a copy of the input DataFrame to avoid modifying the original data
        pos_cash_processed = pos_cash.copy()
        # 1. MONTHS_IN_BALANCE: Number of months reported for each loan
        pos_cash_processed['MONTHS_IN_BALANCE'] = pos_cash_processed.groupby('SK_ID_PREV')['MONTHS_BALANCE'].transform('count')

        # 2. COMPLETED_INSTALLMENTS and ACTIVE_INSTALLMENTS: Counts of completed and active installments for each loan
        pos_cash_processed['COMPLETED_INSTALLMENTS'] = pos_cash_processed.groupby('SK_ID_PREV')['NAME_CONTRACT_STATUS'].transform(lambda x: (x == 'Completed').sum())
        pos_cash_processed['ACTIVE_INSTALLMENTS'] = pos_cash_processed.groupby('SK_ID_PREV')['NAME_CONTRACT_STATUS'].transform(lambda x: (x == 'Active').sum())

        # 3. HAS_DPD and HAS_DPD_DEF: Flags for any delinquency and severe delinquency
        pos_cash_processed['HAS_DPD'] = (pos_cash_processed['SK_DPD'] > 0).astype(int)
        pos_cash_processed['HAS_DPD_DEF'] = (pos_cash_processed['SK_DPD_DEF'] > 0).astype(int)

        # Aggregation at SK_ID_CURR level
        # Aggregate features per customer across all loans
        pos_cash_agg = pos_cash_processed.groupby('SK_ID_CURR').agg({
            'MONTHS_BALANCE': ['min', 'max', 'size'],              # Min/Max reporting date and total months reported
            'CNT_INSTALMENT': ['mean', 'sum'],                     # Average and total installments per customer
            'SK_DPD': ['sum', 'max', 'mean'],                      # Total, max, and mean delinquency days
            'MONTHS_IN_BALANCE': ['max'],                          # Max months in balance for longest-tracked loan
            'COMPLETED_INSTALLMENTS': ['max'],                     # Max completed installments for longest-tracked loan
            'ACTIVE_INSTALLMENTS': ['max'],                        # Max active installments for longest-tracked loan
            'HAS_DPD': ['sum', 'mean'],                            # Total and mean delinquency occurrence
            'HAS_DPD_DEF': ['sum', 'mean']                         # Total and mean severe delinquency occurrence
        }).reset_index()

        # Flatten column names and give meaningful names
        pos_cash_agg.columns = ['_'.join(col).upper() if col[1] else col[0].upper() for col in pos_cash_agg.columns]
        pos_cash_agg.rename(columns={
            'MONTHS_BALANCE_MIN': 'MONTHS_BALANCE_MIN_PC',
            'MONTHS_BALANCE_MAX': 'MONTHS_BALANCE_MAX_PC',
            'MONTHS_BALANCE_SIZE': 'MONTHS_BALANCE_SIZE_PC',
            'CNT_INSTALMENT_MEAN': 'CNT_INSTALMENT_MEAN_PC',
            'CNT_INSTALMENT_SUM': 'CNT_INSTALMENT_SUM_PC',
            'SK_DPD_SUM': 'SK_DPD_SUM_PC',
            'SK_DPD_MAX': 'SK_DPD_MAX_PC',
            'SK_DPD_MEAN': 'SK_DPD_MEAN_PC',
            'MONTHS_IN_BALANCE_MAX': 'LONGEST_REPORTED_LOAN_MONTHS',
            'COMPLETED_INSTALLMENTS_MAX': 'MAX_COMPLETED_INSTALLMENTS',
            'ACTIVE_INSTALLMENTS_MAX': 'MAX_ACTIVE_INSTALLMENTS',
            'HAS_DPD_SUM': 'TOTAL_HAS_DPD_PC',
            'HAS_DPD_MEAN': 'MEAN_HAS_DPD_PC',
            'HAS_DPD_DEF_SUM': 'TOTAL_HAS_DPD_DEF_PC',
            'HAS_DPD_DEF_MEAN': 'MEAN_HAS_DPD_DEF_PC'
        }, inplace=True)
        
        return pos_cash_agg
