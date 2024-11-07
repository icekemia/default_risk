
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class InstallmentsPaymentsAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, inst_pay):
        inst_pay_processed = inst_pay.copy()
        # 1. PAYMENT_DIFF: Difference between payment and installment amounts
        inst_pay_processed['PAYMENT_DIFF'] = inst_pay_processed['AMT_PAYMENT'] - inst_pay_processed['AMT_INSTALMENT']

        # 2. DAYS_LATE_OR_EARLY: Difference between payment date and due date
        inst_pay_processed['DAYS_LATE_OR_EARLY'] = inst_pay_processed['DAYS_ENTRY_PAYMENT'] - inst_pay_processed['DAYS_INSTALMENT']

        # 3. IS_LATE_PAYMENT: Binary flag for late payments
        inst_pay_processed['IS_LATE_PAYMENT'] = (inst_pay_processed['DAYS_LATE_OR_EARLY'] > 0).astype(int)

        # 4. PAYMENT_RATIO: Ratio of payment to installment amount
        inst_pay_processed['PAYMENT_RATIO'] = inst_pay_processed['AMT_PAYMENT'] / (inst_pay_processed['AMT_INSTALMENT'] + 1)

        # Aggregations at SK_ID_PREV level
        prev_level_agg_funcs = {
            'IS_LATE_PAYMENT': ['sum', 'mean'],          # Total and mean late payments
            'DAYS_LATE_OR_EARLY': ['max', 'mean'],       # Max and average days late or early
            'PAYMENT_DIFF': ['mean', 'sum'],             # Average and total payment difference
            'PAYMENT_RATIO': ['mean'],                   # Average payment ratio
            'AMT_PAYMENT': ['sum'],                      # Total payment amount
            'AMT_INSTALMENT': ['sum']                    # Total installment amount
        }
        installments_prev_agg = inst_pay_processed.groupby('SK_ID_PREV').agg(prev_level_agg_funcs)
        installments_prev_agg.columns = ['_'.join(col).upper() for col in installments_prev_agg.columns]
        installments_prev_agg.reset_index(inplace=True)

        # Merge aggregated data back to inst_pay_processed for SK_ID_CURR level aggregation
        inst_pay_processed = inst_pay_processed.merge(installments_prev_agg, on='SK_ID_PREV', how='left')

        # Aggregations at SK_ID_CURR level
        curr_level_agg_funcs = {
            'AMT_INSTALMENT': ['sum', 'mean'],               # Total and mean installment amount per customer
            'AMT_PAYMENT': ['sum', 'mean'],                  # Total and mean payment amount per customer
            'DAYS_INSTALMENT': ['mean', 'std'],              # Mean and standard deviation of installment days
            'DAYS_ENTRY_PAYMENT': ['mean', 'std']            # Mean and standard deviation of payment days
        }
        installments_curr_agg = inst_pay_processed.groupby('SK_ID_CURR').agg(curr_level_agg_funcs)
        installments_curr_agg.columns = ['_'.join(col).strip().upper() for col in installments_curr_agg.columns]
        installments_curr_agg.reset_index(inplace=True)

        # Rename columns for clarity
        installments_curr_agg.rename(columns={
            'AMT_INSTALMENT_SUM': 'AMT_INSTALMENT_SUM_I',
            'AMT_INSTALMENT_MEAN': 'AMT_INSTALMENT_MEAN_I',
            'AMT_PAYMENT_SUM': 'AMT_PAYMENT_SUM_I',
            'AMT_PAYMENT_MEAN': 'AMT_PAYMENT_MEAN_I',
            'DAYS_INSTALMENT_MEAN': 'DAYS_INSTALMENT_MEAN_I',
            'DAYS_INSTALMENT_STD': 'DAYS_INSTALMENT_STD_I',
            'DAYS_ENTRY_PAYMENT_MEAN': 'DAYS_ENTRY_PAYMENT_MEAN_I',
            'DAYS_ENTRY_PAYMENT_STD': 'DAYS_ENTRY_PAYMENT_STD_I'
        }, inplace=True)
        return installments_curr_agg
