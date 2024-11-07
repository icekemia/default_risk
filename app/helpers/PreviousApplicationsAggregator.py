
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .helpers import clean_column_name

class PreviousApplicationsAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, prev_app):
         
        prev_app_processed = prev_app.copy()
        # 1. Grouping loan purposes
        common_purposes = ['Repairs', 'Other']  # Only keeping frequently occurring categories explicitly
        prev_app_processed['CASH_LOAN_PURPOSE_GROUPED'] = prev_app_processed['NAME_CASH_LOAN_PURPOSE'].apply(
            lambda x: 'Unknown' if x in ['XAP', 'XNA'] else (x if x in common_purposes else 'Other')
        )

        # 2. Flag for last application per contract
        prev_app_processed['FLAG_LAST_APPL_PER_CONTRACT'] = prev_app_processed['FLAG_LAST_APPL_PER_CONTRACT'].apply(lambda x: 1 if x == 'Y' else 0)

        # 3. Categorical features for encoding
        categorical_features = [
            'NAME_CONTRACT_TYPE', 'CASH_LOAN_PURPOSE_GROUPED', 'NAME_CONTRACT_STATUS', 
            'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_CLIENT_TYPE', 
            'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 
            'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION'
        ]
        prev_app_processed = pd.get_dummies(prev_app_processed, columns=categorical_features, drop_first=False)
        prev_app_processed.columns = [clean_column_name(col) for col in prev_app_processed.columns]

        # 4. Application and credit ratios
        prev_app_processed['AMT_APPLICATION_CREDIT_RATIO'] = prev_app_processed['AMT_APPLICATION'] / (prev_app_processed['AMT_CREDIT'] + 1)
        prev_app_processed['AMT_ANNUITY_CREDIT_RATIO'] = prev_app_processed['AMT_ANNUITY'] / (prev_app_processed['AMT_CREDIT'] + 1)

        # 5. Status indicators
        for col in ['NAME_CONTRACT_STATUS_Refused', 'NAME_CONTRACT_STATUS_Approved', 'NAME_CONTRACT_STATUS_Canceled']:
            if col not in prev_app_processed.columns:
                prev_app_processed[col] = np.nan

        prev_app_processed['IS_REFUSED'] = (prev_app_processed['NAME_CONTRACT_STATUS_Refused'] == 1).astype(int)
        prev_app_processed['IS_APPROVED'] = (prev_app_processed['NAME_CONTRACT_STATUS_Approved'] == 1).astype(int)
        prev_app_processed['IS_CANCELED'] = (prev_app_processed['NAME_CONTRACT_STATUS_Canceled'] == 1).astype(int)

        # Aggregation at SK_ID_CURR level
        agg_funcs = {
            'AMT_APPLICATION': ['mean', 'min', 'max', 'sum', 'std'],  
            'AMT_CREDIT': ['mean', 'min', 'max', 'sum', 'std'],  
            'AMT_ANNUITY': ['mean', 'max'],
            'CNT_PAYMENT': ['mean', 'max', 'min'],
            'DAYS_DECISION': ['mean', 'max', 'min'],
            'AMT_APPLICATION_CREDIT_RATIO': ['mean', 'max', 'std'],
            'AMT_ANNUITY_CREDIT_RATIO': ['mean', 'max', 'std'],
            'IS_REFUSED': ['sum'], 
            'IS_APPROVED': ['sum'], 
            'IS_CANCELED': ['sum'], 
        }

        previous_app_agg = prev_app_processed.groupby('SK_ID_CURR').agg(agg_funcs)
        previous_app_agg.columns = ['_'.join(col).strip() for col in previous_app_agg.columns]
        previous_app_agg.reset_index(inplace=True)

        # Calculated features for application patterns
        previous_app_agg['APPROVAL_RATE'] = previous_app_agg['IS_APPROVED_sum'] / (
            previous_app_agg['IS_REFUSED_sum'] + previous_app_agg['IS_APPROVED_sum'] + previous_app_agg['IS_CANCELED_sum'] + 1)
        previous_app_agg['CANCELED_RATE'] = previous_app_agg['IS_CANCELED_sum'] / (
            previous_app_agg['IS_REFUSED_sum'] + previous_app_agg['IS_APPROVED_sum'] + previous_app_agg['IS_CANCELED_sum'] + 1)
        previous_app_agg['RECENCY_LAST_APPLICATION'] = previous_app_agg['DAYS_DECISION_min']
        previous_app_agg['AVERAGE_APPLICATION_INTERVAL'] = abs(previous_app_agg['DAYS_DECISION_mean']) / (
            previous_app_agg['IS_APPROVED_sum'] + previous_app_agg['IS_REFUSED_sum'] + 1)
        previous_app_agg.drop(columns=['DAYS_DECISION_min', 'DAYS_DECISION_mean'], inplace=True)

        # Final Aggregated DataFrame with Renaming
        previous_app_agg.rename(columns={
            'AMT_APPLICATION_mean': 'AMT_APPLICATION_MEAN_PA',
            'AMT_APPLICATION_min': 'AMT_APPLICATION_MIN_PA',
            'AMT_APPLICATION_max': 'AMT_APPLICATION_MAX_PA',
            'AMT_APPLICATION_sum': 'AMT_APPLICATION_SUM_PA',
            'AMT_APPLICATION_std': 'AMT_APPLICATION_STD_PA',
            'AMT_CREDIT_mean': 'AMT_CREDIT_MEAN_PA',
            'AMT_CREDIT_min': 'AMT_CREDIT_MIN_PA',
            'AMT_CREDIT_max': 'AMT_CREDIT_MAX_PA',
            'AMT_CREDIT_sum': 'AMT_CREDIT_SUM_PA',
            'AMT_CREDIT_std': 'AMT_CREDIT_STD_PA',
            'AMT_ANNUITY_mean': 'AMT_ANNUITY_MEAN_PA',
            'AMT_ANNUITY_max': 'AMT_ANNUITY_MAX_PA',
            'CNT_PAYMENT_mean': 'CNT_PAYMENT_MEAN_PA',
            'CNT_PAYMENT_max': 'CNT_PAYMENT_MAX_PA',
            'CNT_PAYMENT_min': 'CNT_PAYMENT_MIN_PA',
            'AMT_APPLICATION_CREDIT_RATIO_mean': 'AMT_APPLICATION_CREDIT_RATIO_MEAN_PA',
            'AMT_APPLICATION_CREDIT_RATIO_max': 'AMT_APPLICATION_CREDIT_RATIO_MAX_PA',
            'AMT_APPLICATION_CREDIT_RATIO_std': 'AMT_APPLICATION_CREDIT_RATIO_STD_PA',
            'AMT_ANNUITY_CREDIT_RATIO_mean': 'AMT_ANNUITY_CREDIT_RATIO_MEAN_PA',
            'AMT_ANNUITY_CREDIT_RATIO_max': 'AMT_ANNUITY_CREDIT_RATIO_MAX_PA',
            'AMT_ANNUITY_CREDIT_RATIO_std': 'AMT_ANNUITY_CREDIT_RATIO_STD_PA',
            'IS_REFUSED_sum': 'TOTAL_REFUSED_PA',
            'IS_APPROVED_sum': 'TOTAL_APPROVED_PA',
            'IS_CANCELED_sum': 'TOTAL_CANCELED_PA'
        }, inplace=True)

        return previous_app_agg
