__author__ = 'Nakis'

import pandas as pd
import csv
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from RPI_SOS import RankingsCalculator
from itertools import combinations

class ClassificationModels(object):

    def __init__(self, reg_season, tourney, seeds):

        """
        Wrapper for sklearn classification models for Kaggle NCAA Tourney

        """
        self.Ranks = RankingsCalculator(reg_season, tourney, seeds)
        self.Logit = linear_model.LogisticRegression()
        self.SVM = svm.SVC()
        self.RF = ensemble.RandomForestClassifier()
        self.team_features = []

    def train(self):
        X = self.build_matrix()
        scaled_X = preprocessing.scale(X[:, :-1])
        Y = X[:, -1]
        self.Logit.fit(scaled_X, Y)

    def predict(self, season):
        """
        Infile is tourney_seeds.csv - since removed

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
        return self.Logit.predict_proba(preprocessing.scale(np.array(matrix))), index

    def score(self):
        X = self.build_matrix()
        scaled_X = preprocessing.scale(X[:, :-1])
        Y = X[:, -1]
        svm_scores = cross_validation.cross_val_score(self.SVM, scaled_X, Y, cv=10)
        log_scores = cross_validation.cross_val_score(self.Logit, scaled_X, Y, cv=10)
        rf_scores = cross_validation.cross_val_score(self.RF, scaled_X, Y, cv=10)
        return svm_scores, log_scores, rf_scores

    def build_row(self, season, l_team, h_team):
        temp_list = []
        l_dict = self.Ranks.reg_season[season][l_team]
        h_dict = self.Ranks.reg_season[season][h_team]
        temp_list.append(self.Ranks.calc_RPI(season, l_team))
        temp_list.append(l_dict['net_score'])
        temp_list.append(self.Ranks.calc_SOS(season, l_team))
        #temp_list.append(self.)
        temp_list.append(self.Ranks.calc_RPI(season, h_team))
        temp_list.append(h_dict['net_score'])
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
    CM = ClassificationModels('../raw_data/regular_season_results.csv', '../raw_data/tourney_results.csv', '../raw_data/tourney_seeds.csv')
    #L.train()
    #print L.predict('S')
    svm, l, rf =  CM.score()
    print svm
    print l
    print rf