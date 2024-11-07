from flask import Blueprint, Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import pandas as pd
import traceback
import zipfile
from app.helpers import *
from sklearn.pipeline import Pipeline

main = Blueprint("main", __name__)

prediction_model = joblib.load("app/default_prediction_model.pkl")

REQUIRED_FEATURES = ['NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'NAME_TYPE_SUITE',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'REGION_POPULATION_RELATIVE',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'OWN_CAR_AGE',
 'FLAG_MOBIL',
 'FLAG_EMP_PHONE',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_PHONE',
 'FLAG_EMAIL',
 'OCCUPATION_TYPE',
 'CNT_FAM_MEMBERS',
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'WEEKDAY_APPR_PROCESS_START',
 'HOUR_APPR_PROCESS_START',
 'REG_REGION_NOT_LIVE_REGION',
 'REG_REGION_NOT_WORK_REGION',
 'LIVE_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'LIVE_CITY_NOT_WORK_CITY',
 'ORGANIZATION_TYPE',
 'EXT_SOURCE_1',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3',
 'APARTMENTS_AVG',
 'BASEMENTAREA_AVG',
 'YEARS_BEGINEXPLUATATION_AVG',
 'YEARS_BUILD_AVG',
 'COMMONAREA_AVG',
 'ELEVATORS_AVG',
 'ENTRANCES_AVG',
 'FLOORSMAX_AVG',
 'FLOORSMIN_AVG',
 'LANDAREA_AVG',
 'LIVINGAPARTMENTS_AVG',
 'LIVINGAREA_AVG',
 'NONLIVINGAPARTMENTS_AVG',
 'NONLIVINGAREA_AVG',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'DAYS_LAST_PHONE_CHANGE',
 'FLAG_DOCUMENT_2',
 'FLAG_DOCUMENT_3',
 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_6',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_8',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_10',
 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_12',
 'FLAG_DOCUMENT_13',
 'FLAG_DOCUMENT_14',
 'FLAG_DOCUMENT_15',
 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_17',
 'FLAG_DOCUMENT_18',
 'FLAG_DOCUMENT_19',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_21',
 'AMT_REQ_CREDIT_BUREAU_HOUR',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'AMT_REQ_CREDIT_BUREAU_MON',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'IS_PENSIONER',
 'set',
 'DEBT_CREDIT_RATIO_MEAN',
 'DEBT_CREDIT_RATIO_MAX',
 'TOTAL_STATUS_SCORE',
 'ACTIVE_CREDIT_COUNT',
 'RECENT_CREDIT_COUNT',
 'TOTAL_MONTHS_BALANCE',
 'AMT_APPLICATION_MEAN_PA',
 'AMT_APPLICATION_MIN_PA',
 'AMT_APPLICATION_MAX_PA',
 'AMT_APPLICATION_SUM_PA',
 'AMT_APPLICATION_STD_PA',
 'AMT_CREDIT_MEAN_PA',
 'AMT_CREDIT_MIN_PA',
 'AMT_CREDIT_MAX_PA',
 'AMT_CREDIT_SUM_PA',
 'AMT_CREDIT_STD_PA',
 'AMT_ANNUITY_MEAN_PA',
 'AMT_ANNUITY_MAX_PA',
 'CNT_PAYMENT_MEAN_PA',
 'CNT_PAYMENT_MAX_PA',
 'CNT_PAYMENT_MIN_PA',
 'DAYS_DECISION_max',
 'AMT_APPLICATION_CREDIT_RATIO_MEAN_PA',
 'AMT_APPLICATION_CREDIT_RATIO_MAX_PA',
 'AMT_APPLICATION_CREDIT_RATIO_STD_PA',
 'AMT_ANNUITY_CREDIT_RATIO_MEAN_PA',
 'AMT_ANNUITY_CREDIT_RATIO_MAX_PA',
 'AMT_ANNUITY_CREDIT_RATIO_STD_PA',
 'TOTAL_REFUSED_PA',
 'TOTAL_APPROVED_PA',
 'TOTAL_CANCELED_PA',
 'APPROVAL_RATE',
 'CANCELED_RATE',
 'RECENCY_LAST_APPLICATION',
 'AVERAGE_APPLICATION_INTERVAL',
 'AMT_BALANCE_MEAN_CC',
 'AMT_BALANCE_MAX_CC',
 'AMT_BALANCE_MIN_CC',
 'AMT_BALANCE_SUM_CC',
 'AMT_CREDIT_LIMIT_ACTUAL_MEAN_CC',
 'AMT_CREDIT_LIMIT_ACTUAL_MAX_CC',
 'AMT_PAYMENT_CURRENT_SUM_CC',
 'AMT_PAYMENT_CURRENT_MEAN_CC',
 'SK_DPD_MAX_CC',
 'SK_DPD_MEAN_CC',
 'SK_DPD_SUM_CC',
 'CREDIT_UTILIZATION_MEAN_CC',
 'CREDIT_UTILIZATION_MAX_CC',
 'TOTAL_DELINQUENT_MONTHS_CC',
 'ATM_DRAWINGS_RATIO_MEAN_CC',
 'AMT_INSTALMENT_SUM_I',
 'AMT_INSTALMENT_MEAN_I',
 'AMT_PAYMENT_SUM_I',
 'AMT_PAYMENT_MEAN_I',
 'DAYS_INSTALMENT_MEAN_I',
 'DAYS_INSTALMENT_STD_I',
 'DAYS_ENTRY_PAYMENT_MEAN_I',
 'DAYS_ENTRY_PAYMENT_STD_I',
 'MONTHS_BALANCE_MIN_PC',
 'MONTHS_BALANCE_MAX_PC',
 'MONTHS_BALANCE_SIZE_PC',
 'CNT_INSTALMENT_MEAN_PC',
 'CNT_INSTALMENT_SUM_PC',
 'SK_DPD_SUM_PC',
 'SK_DPD_MAX_PC',
 'SK_DPD_MEAN_PC',
 'LONGEST_REPORTED_LOAN_MONTHS',
 'MAX_COMPLETED_INSTALLMENTS',
 'MAX_ACTIVE_INSTALLMENTS',
 'TOTAL_HAS_DPD_PC',
 'MEAN_HAS_DPD_PC',
 'TOTAL_HAS_DPD_DEF_PC',
 'MEAN_HAS_DPD_DEF_PC']

MODEL_FEATURES = ['AMT_CREDIT_MIN_PA',
 'CNT_FAM_MEMBERS',
 'FLAG_DOCUMENT_14',
 'OCCUPATION_TYPE_Private_service_staff',
 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY',
 'ORGANIZATION_TYPE_Religion',
 'CNT_INSTALMENT_MEAN_PC',
 'TOTAL_DELINQUENT_MONTHS_CC',
 'NAME_TYPE_SUITE_Family',
 'ORGANIZATION_TYPE_Legal_Services',
 'TOTAL_CANCELED_PA',
 'SOCIAL_CIRCLE_DEF_RATIO_30_60',
 'DEBT_CREDIT_RATIO_MAX',
 'ORGANIZATION_TYPE_Police',
 'YEARS_ID_PUBLISH',
 'ORGANIZATION_TYPE_Transport_type_3',
 'FLAG_CONT_MOBILE',
 'ORGANIZATION_TYPE_Restaurant',
 'ORGANIZATION_TYPE_Industry_type_8',
 'FLAG_DOCUMENT_5',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'AMT_APPLICATION_STD_PA',
 'ORGANIZATION_TYPE_Emergency',
 'FLAG_DOCUMENT_6',
 'YEARS_LAST_PHONE_CHANGE',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'NAME_TYPE_SUITE_Unaccompanied',
 'ORGANIZATION_TYPE_Government',
 'NAME_TYPE_SUITE_Other_B',
 'NAME_INCOME_TYPE_Maternity_leave',
 'NAME_HOUSING_TYPE_With_parents',
 'AMT_APPLICATION_SUM_PA',
 'NAME_TYPE_SUITE_Spouse_partner',
 'ATM_DRAWINGS_RATIO_MEAN_CC',
 'NAME_EDUCATION_TYPE_Secondary_secondary_special',
 'ORGANIZATION_TYPE_Industry_type_11',
 'YEARS_ENTRY_PAYMENT_MEAN_I',
 'TOTAL_APPROVED_PA',
 'NONLIVINGAREA_AVG',
 'NAME_INCOME_TYPE_Pensioner',
 'NAME_EDUCATION_TYPE_Lower_secondary',
 'SK_DPD_MEAN_PC',
 'FLAG_WORK_PHONE',
 'EXT_SOURCE_MAX',
 'NAME_HOUSING_TYPE_Municipal_apartment',
 'YEARS_EMPLOYED_RATIO',
 'ORGANIZATION_TYPE_Industry_type_1',
 'FLAG_PHONE',
 'CREDIT_INCOME_RATIO',
 'ORGANIZATION_TYPE_School',
 'REG_CITY_NOT_LIVE_CITY',
 'FLAG_DOCUMENT_18',
 'NAME_FAMILY_STATUS_Unknown',
 'TOTAL_REFUSED_PA',
 'ORGANIZATION_TYPE_Insurance',
 'CREDIT_UTILIZATION_MAX_CC',
 'NAME_HOUSING_TYPE_Office_apartment',
 'NAME_EDUCATION_TYPE_Higher_education',
 'AMT_PAYMENT_CURRENT_MEAN_CC',
 'LANDAREA_AVG',
 'ORGANIZATION_TYPE_Realtor',
 'NAME_HOUSING_TYPE_Rented_apartment',
 'MAX_COMPLETED_INSTALLMENTS',
 'SK_DPD_SUM_CC',
 'APPROVAL_RATE',
 'NAME_CONTRACT_TYPE_Revolving_loans',
 'AMT_APPLICATION_MEAN_PA',
 'WEEKDAY_APPR_PROCESS_START_THURSDAY',
 'FLAG_DOCUMENT_2',
 'WEEKDAY_APPR_PROCESS_START_MONDAY',
 'AMT_ANNUITY_CREDIT_RATIO_MAX_PA',
 'NAME_HOUSING_TYPE_House_apartment',
 'FLAG_EMAIL',
 'OWN_CAR_AGE',
 'EXT_SOURCE_STD',
 'FLAG_DOCUMENT_8',
 'ORGANIZATION_TYPE_Industry_type_4',
 'AMT_GOODS_PRICE',
 'TOTAL_HAS_DPD_DEF_PC',
 'AMT_CREDIT_MEAN_PA',
 'FLAG_DOCUMENT_4',
 'ORGANIZATION_TYPE_Industry_type_13',
 'OCCUPATION_TYPE_IT_staff',
 'ORGANIZATION_TYPE_Culture',
 'REG_REGION_NOT_WORK_REGION',
 'ORGANIZATION_TYPE_University',
 'ORGANIZATION_TYPE_Industry_type_10',
 'ENTRANCES_AVG',
 'ORGANIZATION_TYPE_Trade_type_1',
 'CNT_INSTALMENT_SUM_PC',
 'EXT_SOURCE_2',
 'NAME_EDUCATION_TYPE_Incomplete_higher',
 'FLAG_DOCUMENT_20',
 'TOTAL_HAS_DPD_PC',
 'AMT_ANNUITY',
 'OCCUPATION_TYPE_Realty_agents',
 'OCCUPATION_TYPE_Secretaries',
 'AMT_INCOME_TOTAL',
 'REGION_RATING_CLIENT_W_CITY',
 'ORGANIZATION_TYPE_Mobile',
 'AMT_APPLICATION_CREDIT_RATIO_MAX_PA',
 'LONGEST_REPORTED_LOAN_MONTHS',
 'REG_REGION_NOT_LIVE_REGION',
 'CNT_CHILDREN',
 'NAME_FAMILY_STATUS_Married',
 'EMPLOYED_TO_BIRTH_RATIO',
 'GOODS_INCOME_RATIO',
 'ORGANIZATION_TYPE_Industry_type_6',
 'CHILDREN_RATIO',
 'NAME_INCOME_TYPE_Working',
 'CODE_GENDER_M',
 'FLAG_DOCUMENT_9',
 'AMT_PAYMENT_SUM_I',
 'WEEKDAY_APPR_PROCESS_START_SATURDAY',
 'FLAG_DOCUMENT_16',
 'IS_PENSIONER',
 'OCCUPATION_TYPE_Managers',
 'ORGANIZATION_TYPE_Business_Entity_Type_1',
 'FLAG_DOCUMENT_19',
 'FLAG_OWN_REALTY',
 'CNT_PAYMENT_MAX_PA',
 'AMT_BALANCE_MEAN_CC',
 'AMT_CREDIT_LIMIT_ACTUAL_MEAN_CC',
 'OCCUPATION_TYPE_Drivers',
 'NAME_INCOME_TYPE_State_servant',
 'FLAG_DOCUMENT_21',
 'EXT_SOURCE_1',
 'ORGANIZATION_TYPE_Transport_type_4',
 'MAX_ACTIVE_INSTALLMENTS',
 'ORGANIZATION_TYPE_Construction',
 'AMT_CREDIT_SUM_PA',
 'LIVINGAPARTMENTS_AVG',
 'AMT_APPLICATION_MAX_PA',
 'YEARS_INSTALMENT_MEAN_I',
 'AMT_INSTALMENT_MEAN_I',
 'WEEKDAY_APPR_PROCESS_START_SUNDAY',
 'OCCUPATION_TYPE_Security_staff',
 'AMT_APPLICATION_CREDIT_RATIO_STD_PA',
 'EXT_SOURCE_3',
 'AMT_BALANCE_MIN_CC',
 'ORGANIZATION_TYPE_Industry_type_9',
 'YEARS_REGISTRATION',
 'CANCELED_RATE',
 'CREDIT_TERM',
 'ORGANIZATION_TYPE_Cleaning',
 'FLAG_DOCUMENT_11',
 'ORGANIZATION_TYPE_Industry_type_5',
 'AMT_BALANCE_SUM_CC',
 'INCOME_PER_PERSON',
 'FLAG_OWN_CAR',
 'OCCUPATION_TYPE_Cleaning_staff',
 'EXT_SOURCE_MIN',
 'ORGANIZATION_TYPE_Business_Entity_Type_2',
 'ORGANIZATION_TYPE_Transport_type_1',
 'ORGANIZATION_TYPE_Industry_type_3',
 'SOCIAL_CIRCLE_OBS_RATIO_30_60',
 'ORGANIZATION_TYPE_Business_Entity_Type_3',
 'FLOORSMAX_AVG',
 'OCCUPATION_TYPE_Core_staff',
 'OCCUPATION_TYPE_Laborers',
 'YEARS_BEGINEXPLUATATION_AVG',
 'AMT_ANNUITY_CREDIT_RATIO_MEAN_PA',
 'SK_DPD_MAX_CC',
 'OCCUPATION_TYPE_Medicine_staff',
 'ORGANIZATION_TYPE_XNA',
 'AMT_ANNUITY_MEAN_PA',
 'ORGANIZATION_TYPE_Postal',
 'NAME_FAMILY_STATUS_Single_not_married',
 'ORGANIZATION_TYPE_Housing',
 'AMT_CREDIT',
 'FLAG_DOCUMENT_13',
 'ORGANIZATION_TYPE_Medicine',
 'ORGANIZATION_TYPE_Electricity',
 'FLAG_DOCUMENT_3',
 'CREDIT_UTILIZATION_MEAN_CC',
 'ORGANIZATION_TYPE_Security',
 'AMT_PAYMENT_CURRENT_SUM_CC',
 'OCCUPATION_TYPE_Cooking_staff',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'ORGANIZATION_TYPE_Trade_type_4',
 'LIVE_CITY_NOT_WORK_CITY',
 'AMT_REQ_CREDIT_BUREAU_MON',
 'ORGANIZATION_TYPE_Agriculture',
 'ORGANIZATION_TYPE_Other',
 'MONTHS_BALANCE_SIZE_PC',
 'ORGANIZATION_TYPE_Security_Ministries',
 'ORGANIZATION_TYPE_Hotel',
 'NAME_FAMILY_STATUS_Separated',
 'NAME_FAMILY_STATUS_Widow',
 'OCCUPATION_TYPE_High_skill_tech_staff',
 'ORGANIZATION_TYPE_Trade_type_2',
 'ELEVATORS_AVG',
 'MEAN_HAS_DPD_PC',
 'NAME_INCOME_TYPE_Commercial_associate',
 'OCCUPATION_TYPE_Waiters_barmen_staff',
 'YEARS_EMPLOYED',
 'ORGANIZATION_TYPE_Military',
 'DEBT_CREDIT_RATIO_MEAN',
 'INCOME_EMPLOYED_RATIO',
 'WEEKDAY_APPR_PROCESS_START_TUESDAY',
 'RECENT_CREDIT_COUNT',
 'ORGANIZATION_TYPE_Kindergarten',
 'ORGANIZATION_TYPE_Trade_type_7',
 'FLOORSMIN_AVG',
 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'SK_DPD_SUM_PC',
 'YEARS_ENTRY_PAYMENT_STD_I',
 'REGION_RATING_CLIENT',
 'AGE_INCOME_RATIO',
 'LIVE_REGION_NOT_WORK_REGION',
 'TOTAL_STATUS_SCORE',
 'SK_DPD_MEAN_CC',
 'REGION_POPULATION_RELATIVE',
 'AMT_CREDIT_MAX_PA',
 'AVERAGE_APPLICATION_INTERVAL',
 'YEARS_BUILD_AVG',
 'BASEMENTAREA_AVG',
 'ACTIVE_CREDIT_COUNT',
 'FLAG_DOCUMENT_7',
 'ORGANIZATION_TYPE_Industry_type_12',
 'ORGANIZATION_TYPE_Transport_type_2',
 'YEARS_DECISION_max',
 'CREDIT_GOODS_RATIO',
 'NAME_TYPE_SUITE_Group_of_people',
 'MONTHS_BALANCE_MIN_PC',
 'ORGANIZATION_TYPE_Trade_type_3',
 'AMT_INSTALMENT_SUM_I',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'YEARS_BIRTH',
 'YEARS_INSTALMENT_STD_I',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'EXT_SOURCE_AVG',
 'CNT_PAYMENT_MEAN_PA',
 'FLAG_EMP_PHONE',
 'HOUR_APPR_PROCESS_START',
 'AMT_REQ_CREDIT_BUREAU_HOUR',
 'NAME_TYPE_SUITE_Other_A',
 'FLAG_DOCUMENT_17',
 'TOTAL_MONTHS_BALANCE',
 'ORGANIZATION_TYPE_Bank',
 'NAME_INCOME_TYPE_Student',
 'AMT_CREDIT_STD_PA',
 'MEAN_HAS_DPD_DEF_PC',
 'NONLIVINGAPARTMENTS_AVG',
 'FLAG_DOCUMENT_15',
 'OCCUPATION_TYPE_Sales_staff',
 'OCCUPATION_TYPE_HR_staff',
 'CNT_PAYMENT_MIN_PA',
 'AMT_ANNUITY_MAX_PA',
 'ORGANIZATION_TYPE_Self_employed',
 'ORGANIZATION_TYPE_Trade_type_6',
 'AMT_CREDIT_LIMIT_ACTUAL_MAX_CC',
 'REG_CITY_NOT_WORK_CITY',
 'AMT_PAYMENT_MEAN_I',
 'ORGANIZATION_TYPE_Industry_type_2',
 'TOTAL_DOCUMENTS_FLAGGED',
 'AMT_BALANCE_MAX_CC',
 'ORGANIZATION_TYPE_Telecom',
 'ORGANIZATION_TYPE_Services',
 'APARTMENTS_AVG',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'FLAG_DOCUMENT_10',
 'SK_DPD_MAX_PC',
 'LIVINGAREA_AVG',
 'FLAG_MOBIL',
 'FLAG_DOCUMENT_12',
 'AMT_APPLICATION_MIN_PA',
 'RECENCY_LAST_APPLICATION',
 'OCCUPATION_TYPE_Low_skill_Laborers',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'AMT_APPLICATION_CREDIT_RATIO_MEAN_PA',
 'ORGANIZATION_TYPE_Industry_type_7',
 'ORGANIZATION_TYPE_Trade_type_5',
 'ANNUITY_INCOME_RATIO',
 'AMT_ANNUITY_CREDIT_RATIO_STD_PA',
 'NAME_INCOME_TYPE_Unemployed',
 'COMMONAREA_AVG',
 'MONTHS_BALANCE_MAX_PC']

shared_feature_types = {'numerical_features': [], 'binary_features': [], 'categorical_features': []}

def extract_and_load_zip(zip_path):
    data_frames = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('/tmp/extracted_data')
        for filename in z.namelist():
            file_path = os.path.join('/tmp/extracted_data', filename)
            if filename.endswith('.csv'):
                # Load CSV files into a dictionary for further processing
                data_frames[filename.replace('.csv', '')] = pd.read_csv(file_path)
    return data_frames

def add_credit_income_ratio(X):
    X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
    return X

def add_annuity_income_ratio(X):
    X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    return X

def add_credit_term(X):
    X['CREDIT_TERM'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    return X

def add_age_income_ratio(X):
    X['AGE_INCOME_RATIO'] = (-X['DAYS_BIRTH'] / 365) / X['AMT_INCOME_TOTAL']
    return X

def add_goods_income_ratio(X):
    X['GOODS_INCOME_RATIO'] = X['AMT_GOODS_PRICE'] / X['AMT_INCOME_TOTAL']
    return X

def add_credit_goods_ratio(X):
    X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    return X

def add_children_ratio(X):
    X['CHILDREN_RATIO'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
    return X

def add_income_per_person(X):
    X['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
    return X

def add_years_employed_ratio(X):
    X['YEARS_EMPLOYED_RATIO'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    return X

def add_employed_to_birth_ratio(X):
    X['EMPLOYED_TO_BIRTH_RATIO'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    return X

def add_income_employed_ratio(X):
    X['INCOME_EMPLOYED_RATIO'] = X['AMT_INCOME_TOTAL'] / (-X['DAYS_EMPLOYED'])
    return X

def add_ext_source_avg(X):
    X['EXT_SOURCE_AVG'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    return X

def add_ext_source_min(X):
    X['EXT_SOURCE_MIN'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    return X

def add_ext_source_max(X):
    X['EXT_SOURCE_MAX'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    return X

def add_ext_source_std(X):
    X['EXT_SOURCE_STD'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1).fillna(0)
    return X

def add_total_documents_flagged(X):
    doc_cols = [col for col in X.columns if col.startswith("FLAG_DOCUMENT")]
    X['TOTAL_DOCUMENTS_FLAGGED'] = X[doc_cols].sum(axis=1)
    return X

def fill_social_circle_counts(X):
    for col in ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
                'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']:
        if col in X.columns:
            X[col].fillna(0, inplace=True)
    return X

def add_social_circle_obs_ratio(X):
    if 'OBS_30_CNT_SOCIAL_CIRCLE' in X.columns and 'OBS_60_CNT_SOCIAL_CIRCLE' in X.columns:
        X['SOCIAL_CIRCLE_OBS_RATIO_30_60'] = X['OBS_30_CNT_SOCIAL_CIRCLE'] / (X['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
    return X

def add_social_circle_def_ratio(X):
    if 'DEF_30_CNT_SOCIAL_CIRCLE' in X.columns and 'DEF_60_CNT_SOCIAL_CIRCLE' in X.columns:
        X['SOCIAL_CIRCLE_DEF_RATIO_30_60'] = X['DEF_30_CNT_SOCIAL_CIRCLE'] / (X['DEF_60_CNT_SOCIAL_CIRCLE'] + 1)
    return X

def replace_inf_with_zero(X):
    X.replace([np.inf, -np.inf], 0, inplace=True)
    return X

feature_engineering_functions = [
    add_credit_income_ratio,
    add_annuity_income_ratio,
    add_credit_term,
    add_age_income_ratio,
    add_goods_income_ratio,
    add_credit_goods_ratio,
    add_children_ratio,
    add_income_per_person,
    add_years_employed_ratio,
    add_employed_to_birth_ratio,
    add_income_employed_ratio,
    add_ext_source_avg,
    add_ext_source_min,
    add_ext_source_max,
    add_ext_source_std,
    add_total_documents_flagged,
    fill_social_circle_counts,
    add_social_circle_obs_ratio,
    add_social_circle_def_ratio,
    replace_inf_with_zero
]


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    try:
        # Check for file in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        zip_file = request.files['file']

        # Check if a file was uploaded (i.e., not empty)
        if zip_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Define a local path for temporary file storage
        # UPLOAD_FOLDER = os.path.join(os.getcwd(), 'temp')
        # os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

        import tempfile

        # Use the system's temporary directory
        temp_dir = tempfile.gettempdir()

        # Save the uploaded zip file
        zip_file = request.files['file']
        # zip_path = "/tmp/uploaded_data.zip"
        # zip_file.save(zip_path)

        zip_path = os.path.join(temp_dir, 'uploaded_data.zip')
        zip_file.save(zip_path)

        # Extract datasets from the zip
        data_frames = extract_and_load_zip(zip_path)

        # Define the required datasets
        required_datasets = [
            'application', 'bureau', 'bureau_balance', 'previous_application',
            'credit_card_balance', 'installments_payments', 'pos_cash_balance'
        ]

        # Check if all required datasets are present
        missing_datasets = [name for name in required_datasets if name not in data_frames]
        if missing_datasets:
            return jsonify({"error": f"Missing required datasets: {', '.join(missing_datasets)}"}), 400


        # Process each dataset with its specific aggregator class
        application_data = data_frames.get('application', pd.DataFrame())
        bureau_data = BureauAggregator().fit_transform((data_frames.get('bureau', pd.DataFrame()), data_frames.get('bureau_balance', pd.DataFrame())))
        previous_applications_data = PreviousApplicationsAggregator().fit_transform(data_frames.get('previous_application', pd.DataFrame()))
        credit_card_balance_data = CreditCardBalanceAggregator().fit_transform(data_frames.get('credit_card_balance', pd.DataFrame()))
        installments_payments_data = InstallmentsPaymentsAggregator().fit_transform(data_frames.get('installments_payments', pd.DataFrame()))
        pos_cash_balance_data = PosCashBalanceAggregator().fit_transform(data_frames.get('pos_cash_balance', pd.DataFrame()))

        # Use DatasetJoiner to combine the datasets into a single DataFrame
        combined_data = DatasetJoiner(
            application_data=application_data,
            bureau_data=bureau_data,
            previous_applications_data=previous_applications_data,
            credit_card_balance_data=credit_card_balance_data,
            installments_payments_data=installments_payments_data,
            pos_cash_balance_data=pos_cash_balance_data
        ).join()

        sk_ids = combined_data['SK_ID_CURR']
        combined_data = combined_data.drop(columns=['SK_ID_CURR'])


        pipeline = Pipeline(steps=[
            ('align_features', FeatureAligner(shared_feature_types, required_columns=REQUIRED_FEATURES)),
            ('data_processing', DataPreprocessing(shared_feature_types, debug=True)),
            ('feature_engineering', FeatureEngineeringTransformer(shared_feature_types, additional_features=feature_engineering_functions)),
            ('preprocessing', PreprocessorWrapper(shared_feature_types)),
            ('realign_features', FeatureAligner(shared_feature_types, required_columns=MODEL_FEATURES)),
        ])

        # Process the combined data through the pipeline
        processed_data = pipeline.fit_transform(combined_data)

        # Ensure the processed data has the required features
        processed_data = processed_data[MODEL_FEATURES]

        # Make predictions
        prediction_prob = prediction_model.predict_proba(processed_data)[:, 1]  # Probability of default
        predictions = (prediction_prob > 0.5).astype(int)  # Binary prediction with threshold 0.5

        # Calculate confidence level
        scores = np.round(prediction_prob, 4)

        response = [
            {
                "SK_ID_CURR": int(sk_id),  # ensure IDs are integers for clarity
                "prediction": int(pred),   # 0 or 1 prediction
                "score": float(score)      # probability score for class 1 (e.g., default risk)
            }
            for sk_id, pred, score in zip(sk_ids, predictions, scores)
        ]

        # Return the prediction and confidence level
        return jsonify({"predictions": response})

    except Exception as e:
        # Print traceback for debugging
        print("Error occurred: ", str(e))
        print(traceback.format_exc())

        # Return a 500 error with the exception message
        return jsonify({"error": "Internal server error", "message": str(e)}), 500