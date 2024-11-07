
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .helpers import convert_days_to_years

class BureauAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bureau, bureau_balance = X

        bureau_processed = bureau.copy()
        bureau_processed['HAS_OVERDUE_CREDIT'] = (bureau_processed['CREDIT_DAY_OVERDUE'] > 0).astype(int)
        bureau_processed['HAS_MAX_OVERDUE'] = (bureau_processed['AMT_CREDIT_MAX_OVERDUE'] > 0).astype(int)
        bureau_processed['HAS_CREDIT_PROLONG'] = (bureau_processed['CNT_CREDIT_PROLONG'] > 0).astype(int)
        bureau_processed['HAS_SUM_OVERDUE'] = (bureau_processed['AMT_CREDIT_SUM_OVERDUE'] > 0).astype(int)
        bureau_processed['HAS_CREDIT_LIMIT'] = (bureau_processed['AMT_CREDIT_SUM_LIMIT'] > 0).astype(int)

        # Step 2: Convert DAYS_CREDIT and DAYS_CREDIT_UPDATE to positive years and bin into categories
        bureau_processed['YEARS_CREDIT'] = convert_days_to_years(bureau_processed, 'DAYS_CREDIT')
        bureau_processed['YEARS_CREDIT_UPDATE'] = convert_days_to_years(bureau_processed, 'DAYS_CREDIT_UPDATE')

        bins_years_credit = [0, 5, 10, 20, np.inf]
        labels_years_credit = ['< 5 years ago', '5-10 years ago', '10-20 years ago', '20+ years ago']
        bureau_processed['YEARS_CREDIT_BINNED'] = pd.cut(bureau_processed['YEARS_CREDIT'], bins=bins_years_credit, labels=labels_years_credit)

        bins_years_credit_update = [0, 1, 3, 5, np.inf]
        labels_years_credit_update = ['< 1 year ago', '1-3 years ago', '3-5 years ago', '5+ years ago']
        bureau_processed['YEARS_CREDIT_UPDATE_BINNED'] = pd.cut(bureau_processed['YEARS_CREDIT_UPDATE'], bins=bins_years_credit_update, labels=labels_years_credit_update)

        # Step 3: Calculate credit ratios
        bureau_processed['DEBT_CREDIT_RATIO'] = bureau_processed['AMT_CREDIT_SUM_DEBT'] / (bureau_processed['AMT_CREDIT_SUM'] + 1)
        bureau_processed['ANNUITY_CREDIT_RATIO'] = bureau_processed['AMT_ANNUITY'] / (bureau_processed['AMT_CREDIT_SUM'] + 1)

        # Step 4: Drop unnecessary features after creating derived metrics
        features_to_drop = [
            'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 
            'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY', 
            'DAYS_CREDIT', 'DAYS_CREDIT_UPDATE'
        ]
        bureau_processed.drop(columns=features_to_drop, inplace=True)

        # Step 5: Additional indicators for active credits and currency type
        bureau_processed['IS_ACTIVE_CREDIT'] = (bureau_processed['CREDIT_ACTIVE'] == 'Active').astype(int)
        bureau_processed['IS_CURRENCY_1'] = (bureau_processed['CREDIT_CURRENCY'] == 'currency 1').astype(int)
        bureau_processed.drop(columns=['CREDIT_CURRENCY'], inplace=True)

        # Step 6: Group and dummy-code CREDIT_TYPE and CREDIT_ACTIVE
        bureau_processed['CREDIT_TYPE_GROUPED'] = bureau_processed['CREDIT_TYPE'].apply(lambda x: x if x in ['Consumer credit', 'Credit Card'] else 'Other')
        bureau_processed = pd.get_dummies(bureau_processed, columns=['CREDIT_TYPE_GROUPED', 'CREDIT_ACTIVE'], prefix=['CREDIT_TYPE', 'CREDIT_ACTIVE'], drop_first=False)
        bureau_processed.drop(columns=['CREDIT_TYPE'], inplace=True)

        # Step 7: Calculate custom delinquency score from STATUS in bureau_balance
        def calculate_status_score(status_series):
            score = 0
            for status in status_series:
                if status == '1':
                    score += 1
                elif status == '2':
                    score += 2 ** 2
                elif status == '3':
                    score += 3 ** 3
                elif status == '4':
                    score += 4 ** 4
                elif status == '5':
                    score += 5 ** 5
            return score

        # Aggregation of bureau_balance by SK_ID_BUREAU with custom status score
        bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
            'STATUS': calculate_status_score,
            'MONTHS_BALANCE': ['min', 'max', 'count']
        }).reset_index()
        bureau_balance_agg.columns = ['SK_ID_BUREAU', 'NEW_STATUS_SCORE', 'MONTHS_BALANCE_MIN', 'MONTHS_BALANCE_MAX', 'MONTHS_BALANCE_COUNT']

        # Merge processed bureau data with bureau_balance aggregation
        bureau_with_balance = bureau_processed.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')

        # Step 8: Flag recent credit activity within the past 12 months
        bureau_with_balance['IS_RECENT_CREDIT'] = (bureau_with_balance['MONTHS_BALANCE_MAX'] >= -12).astype(int)
        bureau_with_balance.drop(columns=['SK_ID_BUREAU'], inplace=True)

        # Final aggregation at SK_ID_CURR level
        bureau_agg = bureau_with_balance.groupby('SK_ID_CURR').agg({
            'DEBT_CREDIT_RATIO': ['mean', 'max'],
            'NEW_STATUS_SCORE': ['sum'],  # Total delinquency score across credits
            'IS_ACTIVE_CREDIT': ['sum'],  # Count of active credits
            'IS_RECENT_CREDIT': ['sum'],  # Count of recent credits
            'MONTHS_BALANCE_COUNT': ['sum'],  # Total months balance
        }).reset_index()

        # Flatten multi-level column names
        bureau_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in bureau_agg.columns]

        # Rename columns for clarity
        bureau_agg.rename(columns={
            'SK_ID_CURR_': 'SK_ID_CURR',
            'DEBT_CREDIT_RATIO_mean': 'DEBT_CREDIT_RATIO_MEAN',
            'DEBT_CREDIT_RATIO_max': 'DEBT_CREDIT_RATIO_MAX',
            'NEW_STATUS_SCORE_sum': 'TOTAL_STATUS_SCORE',
            'IS_ACTIVE_CREDIT_sum': 'ACTIVE_CREDIT_COUNT',
            'IS_RECENT_CREDIT_sum': 'RECENT_CREDIT_COUNT',
            'MONTHS_BALANCE_COUNT_sum': 'TOTAL_MONTHS_BALANCE'
        }, inplace=True)
        return bureau_agg
