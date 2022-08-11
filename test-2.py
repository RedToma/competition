import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgbm
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate

train = pd.read_csv('C:/Users/ji98m/OneDrive/바탕 화면/competition_data/train.csv')
test = pd.read_csv('C:/Users/ji98m/OneDrive/바탕 화면/competition_data/test.csv')
sub = pd.read_csv('C:/Users/ji98m/OneDrive/바탕 화면/competition_data/sample_submission.csv')

#print(train.head())
#print(train.info())
print(train.isnull().sum())
print(test.isnull().sum())
print(train.shape)
print(test.shape)

train = train.drop(['index', 'country'],axis = 1)
test = test.drop(['index', 'country'],axis = 1)

#train = train.dropna(axis=0) #결측값 제거
train = train.fillna(train.mean()) #값 메꾸기
test = test.fillna(test.mean()) #값 메꾸기

print(train.isnull().sum())
print(test.isnull().sum())


#train을 target과 feature로 나눔
train_x = train.drop(['nerdiness'], axis=1)
train_y = train['nerdiness']

scaler=MinMaxScaler()
scaler.fit(train_x)
train_x=scaler.transform(train_x)
# 테스트 데이터도 동일 스케일러로
test=scaler.transform(test)


print(train.shape)
print(test.shape)

answer = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12',
          'Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22',
          'Q23','Q24','Q25','Q26',]

#질문 간 상관관계
correlations = train[answer].corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

"""for col in train[answer]:
    print(sorted(train[col].unique()))"""

#목적함수 생성
def lgbm_cv(learning_rate, num_leaves, max_depth, min_child_weight, colsample_bytree, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    model = lgbm.LGBMClassifier(learning_rate=learning_rate,
                                n_estimators = 300,
                                objective = 'binary',
                                #boosting = 'goss',
                                num_leaves = int(round(num_leaves)),
                                max_depth = int(round(max_depth)),
                                min_child_weight = int(round(min_child_weight)),
                                colsample_bytree = colsample_bytree,
                                feature_fraction = max(min(feature_fraction, 1), 0),
                                bagging_fraction = max(min(bagging_fraction, 1), 0),
                                lambda_l1 = max(lambda_l1, 0),
                                lambda_l2 = max(lambda_l2, 0)
                               )
    scoring = {'roc_auc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, train_x, train_y, cv=5, scoring=scoring)
    auc_score = result["test_roc_auc_score"].mean()
    return auc_score

# 입력값의 탐색 대상 구간
pbounds = {'learning_rate' : (0.0001, 0.9),
           'num_leaves': (300, 600),
           'max_depth': (2, 25),
           'min_child_weight': (30, 100),
           'colsample_bytree': (0, 0.99),
           'feature_fraction': (0.0001, 0.99),
           'bagging_fraction': (0.0001, 0.99),
           'lambda_l1' : (0, 0.99),
           'lambda_l2' : (0, 0.99),
          }
    
#객체 생성
lgbmBO = BayesianOptimization(f = lgbm_cv, pbounds = pbounds, verbose = 2, random_state = 0 )

# 반복적으로 베이지안 최적화 수행
# acq='ei'사용
# xi=0.01 로 exploration의 강도를 조금 높임
lgbmBO.maximize(init_points=5, n_iter = 25, acq='ei', xi=0.01)


fit_lgbm = lgbm.LGBMClassifier(learning_rate=lgbmBO.max['params']['learning_rate'],
                               num_leaves = int(round(lgbmBO.max['params']['num_leaves'])),
                               max_depth = int(round(lgbmBO.max['params']['max_depth'])),
                               min_child_weight = int(round(lgbmBO.max['params']['min_child_weight'])),
                               colsample_bytree=lgbmBO.max['params']['colsample_bytree'],
                               feature_fraction = max(min(lgbmBO.max['params']['feature_fraction'], 1), 0),
                               bagging_fraction = max(min(lgbmBO.max['params']['bagging_fraction'], 1), 0),
                               lambda_l1 = lgbmBO.max['params']['lambda_l1'],
                               lambda_l2 = lgbmBO.max['params']['lambda_l2']
                               )

model = fit_lgbm.fit(train_x,train_y)

print(sub)

pred = model.predict(test)

sub["nerdiness"] = pred

print(sub)

sub.to_csv("result.csv", index = False)

print("{}".format(model.score(train_x,train_y)))
print(lgbmBO.max)



"""for col in train[answer]:
    print(sorted(train[col].unique()))
    
flipping_columns = ["Q6", "Q14", "Q18", "Q21", "Q22", "Q23", "Q24", "Q25","Q26"]
for flip in flipping_columns: 
    train[flip] = 6 - train[flip]
    
correlations = train[answer].corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)"""


#특성 분석
"""print(train.introelapse.value_counts()) #초 단위?
print(train.testelapse.value_counts()) #초 단위?
print(train.surveyelapse.value_counts()) #초 단위?
print(train.education.value_counts()) #교육 수준 1,2,3,4 1=Less than high school, 2=High school, 3=University degree, 4=Graduate degree, 0=무응답
print(train.urban.value_counts()) #유년기 거주구역 1=Rural (country side), 2=Suburban, 3=Urban (town, city), 0=무응답
print(train.gender.value_counts()) #성별 1,2,3
print(train.engnat.value_counts()) #모국어가 영어 1,2
print(train.age.value_counts()) #나이 이상치 336발견
print(train.hand.value_counts()) #1,2,3 ex)왼,오른,양손
print(train.married.value_counts()) #혼인 상태 1,2,3 1=Never married, 2=Currently married, 3=Previously married, 0=Other
print(train.familysize.value_counts()) #형제자매 수 이상치 2919발견
print(train.ASD.value_counts()) #자폐 1,2"""