__author__ = 'Nakis'

import pandas as pd
import csv
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from RPI_SOS import RankingsCalculator
from itertools import combinations

class ClassificationModels(object):

    def __init__(self, reg_season, tourney, seeds, power_rankings):

        """
        Wrapper for sklearn classification models for Kaggle NCAA Tourney

        """
        self.Ranks = RankingsCalculator(reg_season, tourney, seeds, power_rankings)
        self.Logit = linear_model.LogisticRegression()
        self.SVM = svm.SVC(probability=True)
        self.GBT = ensemble.GradientBoostingClassifier()
        self.team_features = []

    def train(self):
        X = self.build_matrix()
        scaled_X = preprocessing.scale(X[:, :-1])
        Y = X[:, -1]
        self.Logit.fit(scaled_X, Y)
        self.SVM.fit(scaled_X, Y)
        self.GBT.fit(scaled_X, Y)

    def predict(self, season, a=1, b=1, c=1):
        """
        Output is predictions, index
        """
        #team_list = []
        team_list = self.Ranks.tourney_seeds[season].keys()
        matrix = []
        index = []
        #with open(infile, 'r') as f:
        #    reader = csv.reader(f)
        #    for row in reader:
        #        if row[0] == season:
        #            team_list.append(row[2])
        #            ####MODIFY LATER FOR SEEDS
        #I unindented this whole chunk
        team_list.sort()
        for (l_team, h_team) in combinations(team_list, 2):
            matrix.append(self.build_row(season, l_team, h_team))
            index.append((l_team, h_team))
        matrix = np.array(matrix)
        imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer.fit(matrix)
        logit = self.Logit.predict_proba(preprocessing.scale(matrix))
        SVM = self.SVM.predict_proba(preprocessing.scale(imputer.transform(matrix)))
        GBT = self.GBT.predict_proba(preprocessing.scale(matrix))
        pred = []
        for i in range(len(logit)):
            pred.append((a*logit[i][1] + b*SVM[i][1] + c*GBT[i][1])/(a+b+c))
        return pd.DataFrame(pred, index=index)


    def score(self):
        X = self.build_matrix()
        scaled_X = preprocessing.scale(X[:, :-1])
        imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer.fit(scaled_X)
        scaled_X = imputer.transform(scaled_X)
        Y = X[:, -1]
        svm_scores = cross_validation.cross_val_score(self.SVM, scaled_X, Y, cv=10)
        log_scores = cross_validation.cross_val_score(self.Logit, scaled_X, Y, cv=10)
        gbt_scores = cross_validation.cross_val_score(self.GBT, scaled_X, Y, cv=10)
        return svm_scores.mean(), log_scores.mean(), gbt_scores.mean()

    def build_row(self, season, l_team, h_team):
        temp_list = []
        l_dict = self.Ranks.reg_season[season][l_team]
        h_dict = self.Ranks.reg_season[season][h_team]
        temp_list.append(self.Ranks.calc_RPI(season, l_team))
        temp_list.append(l_dict['net_score'])
        try:
            temp_list.append(l_dict['power'])
        except KeyError:
            temp_list.append(np.NaN)
        temp_list.append(self.Ranks.calc_SOS(season, l_team))
        temp_list.append(self.Ranks.calc_RPI(season, h_team))
        temp_list.append(h_dict['net_score'])
        try:
            temp_list.append(l_dict['power'])
        except KeyError:
            temp_list.append(np.NaN)
        temp_list.append(self.Ranks.calc_SOS(season, h_team))

        #additions
        temp_list += self.Ranks.tourney_seeds[season][l_team]
        temp_list += self.Ranks.tourney_seeds[season][h_team]
        return temp_list

    def build_matrix(self, exclude_season=None):
        output_list = []
        seasons = [x for x in self.Ranks.tourney if x != exclude_season]
        for season in seasons:
            for (l_team, h_team) in self.Ranks.tourney[season]:
                temp_list = self.build_row(season, l_team, h_team)
                temp_list.append(self.Ranks.tourney[season][(l_team, h_team)])
                output_list.append(temp_list)
        return np.array(output_list)


if __name__ == '__main__':
    CM = ClassificationModels('../raw_data/regular_season_results.csv',
                              '../raw_data/tourney_results.csv',
                              '../raw_data/tourney_seeds.csv',
                              '../raw_data/sagp_weekly_ratings.csv')
    #L.train()
    #print L.predict('S')
    svm, l, gbt = CM.score()
    print svm
    print l
    print gbt
