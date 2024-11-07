import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DatasetJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 application_data, 
                 bureau_data=None, 
                 previous_applications_data=None, 
                 credit_card_balance_data=None, 
                 installments_payments_data=None, 
                 pos_cash_balance_data=None):
        self.application_data = application_data
        self.bureau_data = bureau_data
        self.previous_applications_data = previous_applications_data
        self.credit_card_balance_data = credit_card_balance_data
        self.installments_payments_data = installments_payments_data
        self.pos_cash_balance_data = pos_cash_balance_data

    def fit(self, X=None, y=None):
        return self

    def join(self):
        # Start with the main application data
        combined = self.application_data.copy()

        # Merge each dataset if it exists
        if self.bureau_data is not None:
            combined = combined.merge(self.bureau_data, on='SK_ID_CURR', how='left')

        if self.previous_applications_data is not None:
            combined = combined.merge(self.previous_applications_data, on='SK_ID_CURR', how='left')

        if self.credit_card_balance_data is not None:
            combined = combined.merge(self.credit_card_balance_data, on='SK_ID_CURR', how='left')

        if self.installments_payments_data is not None:
            combined = combined.merge(self.installments_payments_data, on='SK_ID_CURR', how='left')

        if self.pos_cash_balance_data is not None:
            combined = combined.merge(self.pos_cash_balance_data, on='SK_ID_CURR', how='left')

        return combined