from sklearn.preprocessing import LabelEncoder
from oliver_util_package import log_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

logger = log_utils.logging.getLogger()

# to see all data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 450)
pd.set_option('display.colheader_justify', 'left')

"""
months_as_customer : int 보험금 납입 기간(누적, 개월)
age : int 연령
policy_number : int 보험 가입번호
policy_bind_date : str 보험 계약일
policy_state : str 보험 계약자 주
policy_csl : 보험한도 : str 사고 발생 시 1인당 보험금 한도액 / 전체 보험금 한도액 (단위 : 천달러)
policy_deductable : int 보험의 자기부담금
policy_annual_premium : float 연간보험료
umbrella_limit : int 보험금 한도
insured_zip : int피보험자 우편번호
insured_sex : str피보험자 성별
insured_education_level :str 피보험자 교육수준
insured_occupation insured_hobbies :str 피보험자의 취미
insured_relationship :str 피보험자 와 보험계약자의 관계
capital-gains : int 자본이득
capital-loss : int 자본손실
incident_date : str 보험 사고 일자
incident_type : str 보험 사고의 종류
collision_type : str 충돌 유형
incident_severity : str 손상정도
authorities_contacted : str 보험사고 발생 당시 연락기관(경찰서, 소방서 등)
incident_state : str 사고 발생 지역
incident_city : str 사고 발생 도시
incident_location : str 사고 발생 위치
incident_hour_of_the_day : int 사고 발생 시각 (예: 20시 경)
number_of_vehicles_involved : int 총 사고 발생 차량 수
property_damage : str 재산 피해 여부
bodily_injuries : int 신체 피해 정도( 0 : 없음, 1: 경상 2: 중상)
witnesses : int 사고 목격자 수
police_report_available : str 경찰 보고서 확인 가능 여부
total_claim_amount : int 총 보험금 청구금액
injury_claim : int 상해 청구금액
property_claim :int 재산 피해 청구 금액
vehicle_claim :int 차량 피해 청구금액
auto_make :str 자동차 제조사
auto_model :str 자동차 모델
auto_year :int 자동차 연식
fraud_reported :str 허위 청구여부 (허위청구 : Y / 정상청구 : N)
"""
file_path = './Dataset/insurance_claims.csv'

# df = pd.read_csv("https://raw.githubusercontent.com/fintech-data/Revolution2/main/data/insurance_claims.csv", index_col=0)
# df.to_csv(file_path)
df = pd.read_csv(file_path)

# logger.info(df.count())
# logger.info(df)
# logger.info(df.info())
# logger.info(df.shape) # 1000, 38
# logger.info(df.isna.__sizeof__()) # na : 48
# logger.info(df.dtypes)
# logger.info(df.nunique())
# logger.info(df.isnull().sum().sum())
# logger.info(df['fraud_reported'].value_counts())
# logger.debug(df[df['fraud_reported'] == 1].total_claim_amount.describe())
# logger.debug(df[df['fraud_reported'] == 0].total_claim_amount.describe())


df = df.replace({'fraud_reported': 'Y'}, 1)  # fraud = 1
df = df.replace({'fraud_reported': 'N'}, 0)  # non_fraud = 0


# series의 데이터타입이 object면 라벨인코더 적용
def object_to_int(dataframe_series: pd.Series):
    if dataframe_series.dtypes == 'object':
        # logger.debug("dataframe_series is object!", dataframe_series.dtypes)
        encoder = LabelEncoder()
        dataframe_series = encoder.fit_transform(dataframe_series)
    return dataframe_series


def distplot(feature, frame, color='r'):
    fig = plt.figure(figsize=(8, 3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color=color)


for feat in df:
    if (df[feat].dtypes == object) & (feat != 'fraud_reported'):
        df[feat] = object_to_int(df[feat])
        # logger.debug(f'{feat} is object type : {df[feat].dtypes}')

    # StdScaler 미적용 bar chart
    # distplot(feat, df)

    if feat != 'fraud_reported':
        df[feat] = StandardScaler().fit_transform(df[feat].values.reshape(-1, 1))

    # StandardScaler : mean : 0 / std : 1
    # StdScaler 적용 bar chart
    # distplot(feat, df, color='c')

# heatmap plt
# plt.figure(figsize=(20, 20))
# df_correlation_matrix = df.corr(method='pearson')
# sns.heatmap(df_correlation_matrix, vmax=0.8, square=True, linewidths=0.1, annot=True)
# plt.show()

df_x = df.drop(
    ['fraud_reported', 'insured_education_level', 'insured_occupation', 'policy_bind_date', 'incident_location',
     'incident_hour_of_the_day', 'auto_model', 'auto_year', 'policy_number', 'insured_zip'], axis=1)
df_y = df['fraud_reported']

# logger.info(df.corr().to_csv('./Dataset/insurance_claims_correlation.csv'))

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, stratify=df_y, shuffle=True,
                                                    random_state=0)
logger.debug(f'X_train {X_train.shape} X_test {X_test.shape} y_tran {y_train.shape} y_test {y_test.shape}')

"""
# KNeighborsClassifier ( N = 5 )
classifier_name = 'KNeighborsClassifier'
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
"""

"""
# SVC
classifier_name = 'SVC'
model = SVC()
model.fit(X_train, y_train)
"""

"""
# RandomForestClassifier
classifier_name = 'RandomForestClassifier'
model = RandomForestClassifier(max_depth=4)
model.fit(X_train, y_train)
"""

# LogisticRegression 
# solver = liblinear (good for small dataset & L1, L2 both support 
# solver = sag(L1 only), saga(both) ( based on SGD - 확률적경사하강법 ) for big dataset
# solver = lbfgs ( lbfgs performance for multi class classification MDL & L2 only ) 
# penalty = L1 or L2 (regulation setting)
# C = cost function(hyperparameter) kinda L1 or L2 (C가 클수록 낮은 강도, 낮을수록 높은 강도 제약조건 설정)
# C 는 계산되는 기울기들을 0 쪽으로 잡아두는 정도를 의미, 즉 C가 높을수록 데이터가 편향 
# class_weight - default : None, weight injection to data directly as hyperparameter
classifier_name = 'LogisticRegression'
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(X_train, y_train)

# for LogisticRegression
y_rfpred_prob = model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Random Forest', color="r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve', fontsize=16)
plt.show()

""" 
# DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, criterion='entropy')
model.fit(X_train, y_train)
"""

"""
# XGBClassifier
model = GradientBoostingClassifier(max_depth=4)
model.fit(X_train, y_train)
"""
lr_label = model.predict(X_test)
scores = model.score(X_test, lr_label)
conf_m = confusion_matrix(y_test, lr_label)
report = classification_report(y_test, lr_label)

lr_accuracy_score = accuracy_score(y_test, lr_label)
lr_precision_score = precision_score(y_test, lr_label)
lr_recall_score = recall_score(y_test, lr_label)
lr_f1_score = f1_score(y_test, lr_label)

logger.debug(f'{classifier_name} accuracy : {lr_accuracy_score}')
logger.debug(f'{classifier_name} precision : {lr_precision_score}')
logger.debug(f'{classifier_name} recall : {lr_recall_score}')
logger.debug(f'{classifier_name} f1 score : {lr_f1_score}')
logger.debug(f'{classifier_name} model score : {scores}')
logger.debug(f'{classifier_name} confusion matrix : \n {conf_m}')
logger.debug(f'{classifier_name} classification_report : \n {report}')
logger.info()
logger.error()

"""
# for RANDOM FOREST
plt.figure(figsize=(4, 3))
sns.heatmap(confusion_matrix(y_test, lr_label), annot=True, fmt="d", linecolor="k", linewidths=3)
plt.title(" RANDOM FOREST CONFUSION MATRIX", fontsize=14)
plt.show()
"""
