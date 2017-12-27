import argparse
import csv
import itertools
from collections import OrderedDict
import numpy as np

y_column = ['overall_rating']
ignore_columns = ['potential','date','player_api_id','player_name','id','preferred_foot','defensive_work_rate','attacking_work_rate']

if __name__ == '__main__':
    filename = 'player_attributes.csv'
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        perf = {'high': 3, 'medium': 2, 'low': 1}
        for row in reader:
            if row['defensive_work_rate'] in perf and row['attacking_work_rate'] in perf:
                sorted_row = OrderedDict(sorted(row.items(),
                  key=lambda item: reader.fieldnames.index(item[0])))
                x_row = []
                y_val = None
                for (i,v) in enumerate(sorted_row):
                    if v in y_column:
                        y_val = float(row[v])
                    elif v not in ignore_columns:
                        a = row[v]
                        if a in perf:
                            a = perf.get(a)
                        x_row.append(a)
                if '2016' not in row['date']:
                    train_row = map(float,x_row)
                    train_row.append(y_val)
                    with open('training_player_data','a') as f:
                        for item in train_row:
                            f.write("%s " % item)
                        f.write("\n")
                else:
                    test_row = map(float,x_row)
                    test_row.append(y_val)
                    with open('testing_player_data','a') as f:
                        for item in test_row:
                            f.write("%s " % item)
                        f.write("\n")
