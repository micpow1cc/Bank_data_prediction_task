import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from statsmodels.stats.proportion import proportions_ztest
# opcja co pozwala wyswietlic caly df
pd.set_option('expand_frame_repr', False)
def load_and_preparing():
    global df, df_campaign
    # zaladowanie danych
    df = pd.read_csv('bank_data_prediction_task.csv')
    # ustawianie typow zmiennych wedlug opisu
    df['job'] = df['job'].astype('category')
    df['marital'] = df['marital'].astype('category')
    df['education'] = df['education'].astype('category')
    df['default'] = df['default'].astype('category')
    df['housing'] = df['housing'].astype('category')
    df['loan'] = df['loan'].astype('category')
    df['test_control_flag'] = df['test_control_flag'].astype('category')
    df['y'] = df['y'].astype('category')
    # zmiana yes no : 1 0 , by wykreslic krzywa ROC
    df['y'].replace(to_replace="no", value=0, inplace=True)
    df['y'].replace(to_replace="yes", value=1, inplace=True)
    df['contact'] = df['contact'].astype('category')
    df['month'] = df['month'].astype('category')
    df['day_of_week'] = df['day_of_week'].astype('category')
    df['duration'] = df['duration'].astype('float64')
    df['campaign'] = df['campaign'].astype('float64')
    df['pdays'] = df['pdays'].astype('int64')
    df['precious'] = df['previous'].astype('float64')
    df['poutcome'] = df['poutcome'].astype('category')
    pdays_values = df['pdays'].value_counts()
    # usuniecie kolumn na potrzebe predykcji na caly dataset
    df.drop(['contact', 'month', 'duration', 'campaign', 'pdays'], axis=1)

    df_control, df_campaign = [x for _, x in df.groupby(df['test_control_flag'] == "campaign group")]
    #brakujące vartosci w kolumnie cons.price.idx zastąpiono średnią
    mean_value = df['cons.price.idx'].mean()
    df['cons.price.idx'].fillna(value=mean_value,inplace=True)
def effectiveness_of_last_campaign():
    df_campaign_y_values = df_campaign['y'].value_counts()
    print(df_campaign_y_values)
    print("skutecznosc kampanii wynosiła:", 2484 / 24712 * 100, "%")
# rf dla kampanii
def randomforest_model():
    global df_campaign, train_df_campaign, rf,profile, cv,y_pred,test_df_campaign,test_target,parameters, train_target
    # najpierw na całym zbiorze danych najważniejsze
    # atrybuty przez które ktos wezmie udział w ,
    # kampanii. A potem na caampaign group o te inne
    # atrybuty
    # zamiana zmiennych category na binarne
    y_column = pd.DataFrame((df_campaign["y"]))
    # usuwanie kolumn mogących źle wplynąć na ACC
    profile = df_campaign
    df_campaign = df_campaign.drop(columns=['y', 'Unnamed: 0', 'pdays'])

    #'y', 'Unnamed: 0', 'pdays' TPR - 60.9 %
    df_campaign = pd.get_dummies(df_campaign)
    df_campaign = pd.concat([df_campaign, y_column], axis=1)
    # podział zmiennych i uczenie modelu
    target = df_campaign['y']
    df_campaign = df_campaign.drop('y', axis=1)
    train_df_campaign, test_df_campaign, train_target, test_target = train_test_split(df_campaign, target,
                                                                                      test_size=0.25, random_state=42)
    # print('Training Atributes Shape:', train_df_campaign.shape)
    # print('Training Targets Shape:', train_target.shape)
    # print('Testing Atributes Shape:', test_df_campaign.shape)
    # print('Testing Targets Shape:', test_target.shape)
    # trenowanie lasu losowego
    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None]

    }
    # parametry n_estimators = 50 i max_depth wybrane przy pomocy metody walidacji krzyżowej GridSearchCV
    rf = RandomForestClassifier(criterion='gini', n_estimators=50, random_state=1,max_depth=50)
    rf.fit(train_df_campaign, train_target)

    #cv = GridSearchCV(rf, parameters, cv=5)
    #cv.fit(train_df_campaign, train_target.values.ravel())

    y_pred = rf.predict(test_df_campaign)
    #conf_mat = confusion_matrix(test_target, y_pred,labels=["test","prediction"])
    print("RandomForest1 results for campaign:")

    print('Accuracy: %.3f' % accuracy_score(test_target, y_pred))
    #print(conf_mat)
    tn, fp, fn, tp = confusion_matrix(test_target,y_pred).ravel()
    print("TN ",tn," FP ",fp," FN",fn," TP ",tp)
    print("True positive rate:",round(tp/(tp+fp),3)*100,"%")
    print("True negative rate:",round(tn/(tn+fn),3)*100,"%") # f # dla kampanii

def importances_plot(classifier,train_df,zbior_danych,nazwa):
    train_df = pd.DataFrame(
        train_df)  # to zrobione po to bo by wykreslic importance of variables musi byc dataframe pandas
    importances = classifier.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    plt.title(f'feature importances for {nazwa}')
    plt.bar(range(train_df.shape[1]), importances[sorted_indices], align='center',color="black")
    plt.xticks(range(train_df.shape[1]), zbior_danych.columns[sorted_indices], rotation=90)

    plt.tight_layout()
    plt.show()
def profile_of_person():    # Ktore osoby pytac:
    global profile

    profile_no, profile_yes = [x for _, x in profile.groupby(profile['y'] == 1)]
    age = np.mean(profile_yes.age)
    age_sd = np.std(profile_yes.age)

    #print(profile_yes['job'].value_counts())
    #print(profile_yes['marital'].value_counts())
    #print(profile_yes['education'].value_counts())
    #print(profile_yes['default'].value_counts())
    #print(profile_yes['loan'].value_counts())
    #print(profile_yes['contact'].value_counts())

    print(profile_yes['duration'].value_counts())

    print('His age is:  %.3f' % age,'+/-  %.3f'  % age_sd)
    print('Thier most common job is : admin.,blue-collar,technican')
    print('Thier most common marital status is : high.school,university.degree')
    print('Almost everybody dont have credit in default')
    print('Big majority dont have a loan')
    print('Big majority of them were ellular> telephone')
    print('People wanted to subscribe to term deposit most often in may,july,august')
    print('There is no dependence in what day people want to subscribe to term deposit')

    bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540,600, 660,720,780,840,900,960,1020,1080,1140,3200]
    cat = pd.cut(x=profile_yes['duration'],bins=bins,include_lowest=True)
    plt.title("Duration of talking in seconds")
    plt.xlabel("Seconds [s]")
    plt.ylabel("Quantity")
    ax = cat.value_counts(sort=False).plot.bar(rot=90, color="black", figsize=(30, 20))
    plt.show()


    #job_admin.  job_blue-collar  job_entrepreneur  job_housemaid  job_management  job_retired  job_self-employed  job_services  job_student  job_technician  job_unemployed  job_unknown
def display(results): # metoda pozwalajaca wyswietlic najlepsze parametry max_depth i n_estimators
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
def ROC_curve(classifier,test_df,test_Y,nazwa,classifier2,second_test_df,second_test_Y,nazwa2):
    x = np.linspace(0, 1, 10)
    y = x

    y_pred_proba = classifier.predict_proba(test_df)[::, 1]
    y_pred_proba2 =classifier2.predict_proba(second_test_df)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(test_Y, y_pred_proba)
    fpr2,tpr2,_ = metrics.roc_curve(second_test_Y,y_pred_proba2)
    auc = roc_auc_score(test_Y, y_pred_proba)
    auc2 = roc_auc_score(second_test_Y,y_pred_proba2)
    label = f'{nazwa},AUC=', round(auc, 3)
    label2 =f'{nazwa2},AUC=', round(auc2, 3)
    plt.plot(fpr, tpr, label=label)
    plt.plot(fpr2,tpr2,label=label2)
    plt.plot(x, y, label="Random choosing")

    plt.title('ROC curve')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='upper left')
    plt.show()
# rf2 profil osoby
def Random_forest_2():
    global df,cv,X_train, X_test, Y_train, Y_test,rf2,target
    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None]

    }
    y_column = pd.DataFrame((df["y"]))
    # usuwanie kolumn mogących źle wplynąć na ACC
    df = df.drop(columns=['y', 'Unnamed: 0', 'pdays','contact','day_of_week','duration','campaign','month'])
    df = pd.get_dummies(df)
    df = pd.concat([df, y_column], axis=1)
    # podział zmiennych i uczenie modelu
    target = df['y']
    df = df.drop('y', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(df,target,test_size=0.25, random_state=42)
    rf2 = RandomForestClassifier(criterion='gini', n_estimators=250, random_state=1,max_depth=8)
    #cv = GridSearchCV(rf, parameters, cv=5)
    #cv.fit(X_train, Y_train.values.ravel())
    rf2.fit(X_train,Y_train)
    y_pred = rf2.predict(X_test)
    print("RandomForest2 for all results:")

    print('Accuracy: %.3f' % accuracy_score(Y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    print("TN ", tn, " FP ", fp, " FN", fn, " TP ", tp)
    print("True positive rate:", round(tp / (tp + fp), 3) * 100, "%")
    print("True negative rate:", round(tn / (tn + fn), 3) * 100, "%")
#XGboost dla kampanii
def XGboost():
    global train_df_campaign, test_df_campaign, train_target, test_target,cv, X_train, X_test, Y_train, Y_test,xg_reg,train_xgb,test_xgb,train_xgb_target,test_xgb_target
    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None]

    }
    train_xgb=train_df_campaign
    test_xgb=test_df_campaign
    train_xgb_target=train_target
    test_xgb_target=test_target

    xg_reg = xgb.XGBClassifier(objective='binary:logistic',colsample_bytree = 0.3, max_depth = 4,alpha=10,n_estimators=100)
    # TPR 65.6% dla colsample_bytree = 0.3, max_depth = 5,alpha=10,n_estimators=10
    #cv = GridSearchCV(xg_reg, parameters, cv=5)
    #cv.fit(train_df_campaign, train_target.values.ravel())
    xg_reg.fit(train_xgb,train_xgb_target)

    preds= xg_reg.predict(test_xgb)
    print("XGboostClassifier for campaign results:")
    print('Accuracy: %.3f' % accuracy_score(test_xgb_target, preds))
    tn, fp, fn, tp = confusion_matrix(test_xgb_target, preds).ravel()
    print("TN ", tn, " FP ", fp, " FN", fn, " TP ", tp)
    print("True positive rate:", round(tp / (tp + fp), 3) * 100, "%")
    print("True negative rate:", round(tn / (tn + fn), 3) * 100, "%")

# XGboost2 profil osoby

def XGboost2():
    global train_df_campaign, test_df_campaign, train_target, test_target,cv, X_train, X_test, Y_train, Y_test,xg_reg2,df,X_train2,X_test2,Y_train2,Y_test2,target
    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None]

    }
    target2 = target
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(df,target2,test_size=0.25, random_state=42)
    xg_reg2 = xgb.XGBClassifier(objective='binary:logistic',colsample_bytree = 0.3, max_depth = 4,alpha=10,n_estimators=100)
    # TPR 65.6% dla colsample_bytree = 0.3, max_depth = 5,alpha=10,n_estimators=10
    #cv = GridSearchCV(xg_reg, parameters, cv=5)
    #cv.fit(X_train, Y_train.values.ravel())
    xg_reg2.fit(X_train2,Y_train2)

    preds= xg_reg2.predict(X_test2)
    print("XGboostClassifier for all results:")
    print('Accuracy: %.3f' % accuracy_score(Y_test2, preds))
    tn, fp, fn, tp = confusion_matrix(Y_test2, preds).ravel()
    print("TN ", tn, " FP ", fp, " FN", fn, " TP ", tp)
    print("True positive rate:", round(tp / (tp + fp), 3) * 100, "%")
    print("True negative rate:", round(tn / (tn + fn), 3) * 100, "%")

def does_clasiffier_is_better_than_random():
    significance = 0.05
    # p2=64% , poprzednia kampania p1=50% , wybrano klasyfikator Rf2 z powodu największego współczynnika TPR
    sample_success = 3179 # wartosc sample succes jest wyliczona z proporcji TPR i wartosci z ostatniej kampanii
    # sample_succes = 2484*64/100
    sample_size = 24712
    # Ho -> p1=p2
    # H1 -> p1<p2
    null_hypothesis = 0.50

    stat, p_value = proportions_ztest(count=sample_success, nobs=sample_size, value=null_hypothesis,
                                      alternative='smaller')
    # report
    print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
    if p_value > significance:
        print("przyjmujemy hipoteze zerową")
    else:
        print("odrzucami hipoteze zerowa - sugeruje prawdziwosc hipotezy alternatywnej")
    # biorąc pod uwagę dane z wczesniejszej kampanii, której wydajność wynosi ~10%, wybrany model, może zwiększyć wydajność następnej kampanii do ~12.9%.

if __name__ == "__main__":
    rf_1 = "Random forest for campaign data"
    rf_2 = "Random forest for all data"
    xg_reg_1 = "XGboost for campaign data"
    xg_reg_2 = "XGboost for all data"

    load_and_preparing()
    effectiveness_of_last_campaign()
    randomforest_model()
    profile_of_person()
    Random_forest_2()
    XGboost()
    XGboost2()
    #display(cv)
    ROC_curve(rf2,X_test, Y_test, rf_2,xg_reg2,X_test2,Y_test2,xg_reg_2)
    importances_plot(xg_reg2,X_test2,df,xg_reg_2)
    #does_clasiffier_is_better_than_random()


