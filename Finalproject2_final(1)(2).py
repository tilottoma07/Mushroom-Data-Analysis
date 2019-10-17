import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


#this function read csv file into panda data frame
def read_csv():
    try:
        df = pd.read_csv('mushrooms.csv')
        mushroomdf = pd.DataFrame(df)
        return mushroomdf
    except:
        print('The csv file does not exist. Exiting...')
        exit()

#study the data frame & make a pie plot to check the balance of the dataset
def some_statisics(mushroomdf):
    print('\nSome Basic Info of the Dataset ')
    print(mushroomdf.describe())

    #make a pie plot to understand the percentage of edible and poisonous mushroom
    edible_percent = len(mushroomdf[(mushroomdf['class'] == 'e')]) / len(mushroomdf['class'])
    poisonous_percent = len(mushroomdf[(mushroomdf['class'] == 'p')]) / len(mushroomdf['class'])

    labels = 'edible', 'poisonous'
    sizes = [edible_percent,poisonous_percent]
    explode = (0, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set(title='Percentage of Edible and Poisonous Mushrooms in the dataset')
    plt.show()

#The data is categorial so I used LabelEncoder to convert the each label into numerical equivalent.
def label_encoder(mushroomdf):
    labelEncoder = preprocessing.LabelEncoder()

    for col in mushroomdf.columns:
        mushroomdf[col] = labelEncoder.fit_transform(mushroomdf[col])

    return mushroomdf

#This function splits the dataset into test and train portion
def train_test(mushroomdf):
    # Train Test Split
    X = mushroomdf.drop('class', axis=1)
    y = mushroomdf['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X, X_train, X_test, y, y_train, y_test

#This function uses different Machine Learning Algorithm to model the dataset and compares there accuracy
def ML_Algo_(X_train, X_test, y_train, y_test, flag = False):
    keys = []
    scores = []
    models = {'Logistic Regression': LogisticRegression(),
              'Decision Tree': DecisionTreeClassifier(),
              'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
              'Linear SVM': SVC(kernel='rbf',random_state=42),
              'Random Forest':RandomForestClassifier(n_estimators= 100, n_jobs=-1)}

    for k, v in models.items():
        print('*******************************************')
        mod = v
        mod.fit(X_train, y_train)
        pred = mod.predict(X_test)
        print('Results for: ' + str(k) + '\n')
        print('Classification Report', classification_report(y_test, pred))
        cm = confusion_matrix(y_test, pred)
        acc = accuracy_score(y_test, pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Edible", "Poisonous"],yticklabels=["Edible","Poisonous"])
        if flag == True:
            plt.title('After retraining, Confusion matrix for '+ str(k) +' : Accuracy Score: '+str(acc))
        else:
            plt.title('Confusion matrix for '+ str(k) +' : Accuracy Score: '+str(acc))

        plt.show()

        print('\n' + '\n')
        keys.append(k)
        scores.append(acc)
        table = pd.DataFrame({'model': keys, 'accuracy score': scores})

    print('**********************************************')
    if flag == True:
        print('After retraining, Accuracy Score Summary')
    else:
        print('Accuracy Score Summary')
    print(table)

# As all the models give close to 100% acuracy (could be overfitting), we have veried the depth of the decision tree to check
# whether accuracy veries with the decision tree depth
def testing_decision_tree_depth(X,y):
    accuracy=[]
    depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for i in depth:
        X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)
        clf = DecisionTreeClassifier(max_depth = i).fit(X_train, y_train)
        score=clf.score(X_test,y_test )
        accuracy.append(score)

    fig, ax = plt.subplots()
    ax.plot(depth, accuracy)
    ax.set(xlabel='Depth of tree', ylabel='Accuracy',
       title='Decision Tree - Accuracy vs Depth of Decision tree')
    ax.grid()
    plt.show()

# This function is used to find out the most important features to diffrentiate between edible and poisonous mushrooms
def rfc_Algo_feature(X_train, y_train, mushroomdf):

    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(X_train, y_train)
    X = mushroomdf.drop('class', axis=1)
    importances = rfc.feature_importances_

    print('**********************************************')
    print('Sorted Feature Importance')
    df = pd.DataFrame({'Features':X.columns,'Weight':importances})
    df = df.sort_values(by ='Weight')

    print(df)

    ax = df.plot.bar(x='Features', y='Weight', rot=90)
    ax.set_xlabel('Features',fontsize=20)
    ax.set_ylabel('Importances',fontsize=20)
    ax.set_title('Features Importance',fontsize=22)
    plt.show()

# after finding out the most important features we choose the most important 5 features and retrain the model again
def re_train_data(X_train, X_test, y_train, y_test,mushroomdf):
    X = mushroomdf[['odor','gill-color','gill-size','spore-print-color','ring-type']]
    y = mushroomdf['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(X_train, y_train)
    ML_Algo_(X_train, X_test, y_train, y_test,True)

#This function re-label the classes in a elaborate manner
def elaborate_the_labels(mushroomdf_copy):
    mushroomdf_copy['class'] = mushroomdf_copy['class'].replace({'p':'poisonous','e':'edible'})
    mushroomdf_copy['gill-color'] = mushroomdf_copy['gill-color'].replace({'k':'black','n':'brown','b':'buff','h':'chocolate',
                             'g':'gray', 'r':'green','o':'orange','p':'pink',
                             'u':'purple','e':'red','w':'white','y':'yellow'})
    mushroomdf_copy['spore-print-color'] = mushroomdf_copy['spore-print-color'].replace({'k':'black','n':'brown','b':'buff','h':'chocolate',
                              'r':'green','o':'orange','u':'purple','w':'white','y':'yellow'})
    mushroomdf_copy['odor'] = mushroomdf_copy['odor'].replace({'a':'almond','l':'anise','c':'creosote','y':'fishy',
                                                                'f':'foul','m':'musty','n':'none','p':'pungent','s':'spicy'})
    mushroomdf_copy['gill-size'] = mushroomdf_copy['gill-size'].replace({'b':'broad','n':'narrow'})
    mushroomdf_copy['population'] = mushroomdf_copy['population'].replace({'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several','y':'solitary'})

    return mushroomdf_copy

#Auto-labels the number of mushrooms for each bar color.
def autolabel(ax,rects,fontsize=14):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=fontsize)

# This function makes bar plots for some important features 
def bar_plot_for_important_features(col,mushroomdf_copy,name):
    features = mushroomdf_copy[col].value_counts()
    features_height = features.values.tolist() #Provides numerical values
    features_labels = features.axes[0].tolist() #Converts index labels object to list
    poisonous_od = [] #Poisonous odor list
    edible_od = []    #Edible odor list
    ind = np.arange(len(list(name)))  # the x locations for the groups
    for feature in features_labels:
        size = len(mushroomdf_copy[mushroomdf_copy[col] == feature].index)
        edibles = len(mushroomdf_copy[(mushroomdf_copy[col] == feature) & (mushroomdf_copy['class'] == 'edible')].index)
        edible_od.append(edibles)
        poisonous_od.append(size-edibles)

    #PLOT Preparations and Plotting
    width = 0.40
    fig, ax = plt.subplots(figsize=(12,7))
    edible_bars = ax.bar(ind, edible_od , width, color='blue')
    poison_bars = ax.bar(ind+width, poisonous_od , width, color='red')

    #Add some text for labels, title and axes ticks
    ax.set_xlabel(col,fontsize=20)
    ax.set_ylabel('Count',fontsize=20)
    ax.set_title('Edible and Poisonous Mushrooms Based on "'+ col+'"',fontsize=22)
    ax.set_xticks(ind + width / 2) #Positioning on the x axis
    ax.set_xticklabels(name,fontsize = 12)
    ax.legend((edible_bars,poison_bars),('edible','poisonous'),fontsize=17)
    autolabel(ax,edible_bars)
    autolabel(ax,poison_bars)
    plt.show()


def main():
    mushroomdf = read_csv()
    mushroomdf_copy = mushroomdf.copy()
    some_statisics(mushroomdf)
    mushroomdf = label_encoder(mushroomdf)
    X, X_train, X_test, y, y_train, y_test = train_test(mushroomdf)
    ML_Algo_(X_train, X_test, y_train, y_test)
    testing_decision_tree_depth(X,y)
    rfc_Algo_feature(X, y, mushroomdf)
    re_train_data(X_train, X_test, y_train, y_test, mushroomdf)
    mushroomdf_copy = elaborate_the_labels(mushroomdf_copy)

    bar_plot_for_important_features('odor', mushroomdf_copy,('none', 'foul','fishy','spicy',
                                    'almond','anise','pungent','creosote','musty'))
    bar_plot_for_important_features('gill-color', mushroomdf_copy,('buff','pink','white',
                                    'brown','gray','chocolate','purple','black','red',
                                    'yellow','orange','green'))
    bar_plot_for_important_features('gill-size', mushroomdf_copy,('broad','narrow'))
    bar_plot_for_important_features('spore-print-color', mushroomdf_copy,('white','brown',
                                    'black','chocolate','green','orange','buff','purple','yellow'))
    bar_plot_for_important_features('population', mushroomdf_copy,('several','solitary',
                                    'scattered','numerous','abundant','clustered'))

if __name__=='__main__':
    main()
