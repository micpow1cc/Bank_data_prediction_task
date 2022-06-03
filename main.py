import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score


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
    # print(pdays_values)
    # usuniecie kolumn na potrzebe predykcji na caly dataset
    df.drop(['contact', 'month', 'duration', 'campaign', 'pdays'], axis=1)
    # 16474 ostatni index osoby w campagin group
    # procent osob, ktorym udalo sie zaoferowac lokate
    df_control, df_campaign = [x for _, x in df.groupby(df['test_control_flag'] == "campaign group")]


def effectiveness_of_last_campaign():
    df_campaign_y_values = df_campaign['y'].value_counts()
    print(df_campaign_y_values)
    print("skutecznosc kampanii wynosiła:", 2484 / 24712 * 100, "%")



def randomforest_model():
    global df_campaign, train_df_campaign, rf,profile, cv,y_pred,test_df_campaign,test_target
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
    ns_probs = [0 for _ in range(len(test_target))]

    #cv = GridSearchCV(rf, parameters, cv=5)
    #cv.fit(train_df_campaign, train_target.values.ravel())

    y_pred = rf.predict(test_df_campaign)
    #conf_mat = confusion_matrix(test_target, y_pred,labels=["test","prediction"])
    print('Accuracy: %.3f' % accuracy_score(test_target, y_pred))
    #print(conf_mat)
    tn, fp, fn, tp = confusion_matrix(test_target,y_pred).ravel()
    print("TN ",tn," FP ",fp," FN",fn," TP ",tp)
    print("True positive rate:",round(tp/(tp+fp),3)*100,"%")
    print("True negative rate:",round(tn/(tn+fn),3)*100,"%")
# ACC 86.2%
# TPR 58.8
# TNR 88.3%
#print(y_pred)


def importances_plot():
    global train_df_campaign
    train_df_campaign = pd.DataFrame(
        train_df_campaign)  # to zrobione po to bo by wykreslic importance of variables musi byc dataframe pandas
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    plt.title('Feature Importance')
    plt.bar(range(train_df_campaign.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(train_df_campaign.shape[1]), df_campaign.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()
    """"
    plt.rcdefaults()
    fig,ax = plt.subplots()
    labels = list(train_df_campaign.shape[1])
    y_pos = list(importances)
    ax.barh(y_pos,labels, align='center')
    ax.set_yticks(y_pos,labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Importance of variables')
    plt.show()
    """



def profile_of_person():    # Ktore osoby pytac:

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
    print('The most often people')

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



def ROC_curve():
    x = np.linspace(0, 1, 10)
    y = x

    y_pred_proba = rf.predict_proba(test_df_campaign)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(test_target, y_pred_proba)
    auc = roc_auc_score(test_target, y_pred_proba)
    label = 'RandomForest,ROC AUC=', round(auc, 3)
    plt.plot(fpr, tpr, label=label)
    plt.plot(x, y, label="No skill")
    plt.title('ROC curve')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='upper left')
    plt.show()
def neural_network():
        print("")

load_and_preparing()
effectiveness_of_last_campaign()
randomforest_model()
importances_plot()
profile_of_person()
ROC_curve()
#display(cv)




