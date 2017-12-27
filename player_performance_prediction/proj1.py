import csv
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import numpy
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from collections import OrderedDict
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

y_column = ['overall_rating']
ignore_columns = ['potential','date','player_api_id','player_name','id','preferred_foot','defensive_work_rate','attacking_work_rate']

def load_data(filename):
    X = []
    y = []
    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter=' ')
        for row in reader:
            row = filter(None, row)
            last_ind = len(row)
            if last_ind == 34:
                y.append(float(row[last_ind-1]))
                a = list(map(float,row[0:last_ind-1]))
                X.append(list(map(float,row[0:last_ind-1])))
    return X, y

def load_data_to_predict():
    filename = 'arsenal2017.csv'
    X = []
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sorted_row = OrderedDict(sorted(row.items(),
              key=lambda item: reader.fieldnames.index(item[0])))
            x_row = []
            y_val = None
            for (i,v) in enumerate(sorted_row):
                if v not in ignore_columns:
                    a = row[v]
                    x_row.append(a)
            full_row = map(float,x_row)
            X.append(full_row)
            # with open('data_without_2016','a') as f:
            #     for item in full_row:
            #         f.write("%s " % item)
            #     f.write("\n")
        return X

if __name__ == '__main__':
    X_train, y_train = load_data('training_player_data')
    X_test, y_test = load_data('testing_player_data')
    player_data = load_data_to_predict()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # with open('target_results','w') as f:
    #     for item in y_test:
    #         f.write("%s\n" % item)

    regr = linear_model.LinearRegression()
    lr_scs = cross_validate(regr, X_train, y_train, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
    print('Liner Regression:')
    print(numpy.absolute(lr_scs['test_score']))
    regr.fit(X_train, y_train)
    y_pred_lr = regr.predict(X_test)
    err = mean_squared_error(y_test, y_pred_lr)
    print('Test set Error: '+str(err)+'\n')

    lamb = []
    l = 0.00001
    while l != 10:
        lamb.append(l)
        l*=10

    test_err = OrderedDict({})
    train_err = OrderedDict({})
    l2_norm = OrderedDict({})
    weights = OrderedDict({})
    best_rl = [l, 1000]
    best_ll = [l, 1000]
    lkfold_cv_score = {}
    rkfold_cv_score = {}
    for l in lamb:
        rlr = Ridge(alpha=l,normalize=True)
        llr = Lasso(alpha=l,normalize=True)
        # Perform 5-fold validation
        rscs = cross_validate(rlr, X_train, y_train, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
        lscs = cross_validate(llr, X_train, y_train, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
        rkfold_cvs = numpy.mean(numpy.absolute(rscs['test_score']))
        rkfold_trs = numpy.mean(numpy.absolute(rscs['train_score']))
        rkfold_cv_score.update({l : rkfold_cvs})

        lkfold_cvs = numpy.mean(numpy.absolute(lscs['test_score']))
        lkfold_trs = numpy.mean(numpy.absolute(lscs['train_score']))
        lkfold_cv_score.update({l : lkfold_cvs})

        if best_rl[1] > rkfold_cvs:
            best_rl = [l, rkfold_cvs]
        if best_ll[1] > lkfold_cvs:
            best_ll = [l, lkfold_cvs]

    rlr = Ridge(alpha=best_rl[0],normalize=True)
    rlr.fit(X_train,y_train)
    llr = Lasso(alpha=best_ll[0],normalize=True)
    llr.fit(X_train,y_train)
    # Predict for training data and calculate the RMSE
    y_pred_rlr = rlr.predict(X_test)
    y_pred_llr = llr.predict(X_test)
    rlerr = mean_squared_error(y_test, y_pred_rlr)
    llerr = mean_squared_error(y_test, y_pred_llr)
    print('Ridge regression:')
    print(rkfold_cv_score)
    print(str(rlerr)+'\n')
    print('Lasso regression:')
    print(lkfold_cv_score)
    print(str(llerr)+'\n')

    regressor = DecisionTreeRegressor()
    dt_scs = cross_validate(regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
    print(numpy.absolute(dt_scs['test_score']))
    regressor.fit(X_train,y_train)
    #Predict using Decision Tree Regression
    y_pred_dt = regressor.predict(X_test)
    err = mean_squared_error(y_test, y_pred_dt)
    print('Decision Tree Regression: '+str(err))


    # Training Random Forest Regression Model
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    rf_scs = cross_validate(regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
    print(numpy.absolute(rf_scs['test_score']))
    regressor.fit(X_train, y_train)
    # Predict Result from Random Forest Regression Model
    y_pred_rf = regressor.predict(X_test)
    err = mean_squared_error(y_test, y_pred_rf)

    print(y_pred_rf)
    print('Random Forest Regression: '+str(err))

    player_rankings = regressor.predict(player_data)
    print('Overall Player Ranking:')
    print(player_rankings)

    # midfield = OrderedDict({})
    # goalkeepers = OrderedDict({})
    # defenders = OrderedDict({})
    # attack = OrderedDict({})

    # for i in range(y_pred_rf):
    #     if players['class'] == 'midfield':
    #         midfield.update(({players[i]['name']:y_pred_rf[i]})
    #     elsif players['class'] == 'defender':
    #         defenders.update(({players[i]['name']:y_pred_rf[i]})
    #     else if players['class'] == 'attack':
    #         attack.update(({players[i]['name']:y_pred_rf[i]})
    #     else:
    #         goalkeepers.update(({players[i]['name']:y_pred_rf[i]})


    # import operator
    # sorted_midfield = sorted(midfield.items(), key=operator.itemgetter(1))
    # sorted_attack = sorted(attack.items(), key=operator.itemgetter(1))
    # sorted_defenders = sorted(defenders.items(), key=operator.itemgetter(1))
    # sorted_goalkeepers = sorted(goalkeepers.items(), key=operator.itemgetter(1))
    #
    # print('Playing 11')
    # print(sorted_goalkeepers.values()[0])
    # print(sorted_attack.values()[0:num_attack-1])
    # print(sorted_midfield.values()[0:num_midfield-1])
    # print(sorted_defenders.values()[0:num_defenders-1])
