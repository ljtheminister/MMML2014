__author__ = 'Nakis'

import pandas as pd
import csv
import numpy as np
from sklearn import linear_model, preprocessing
from RPI_SOS import RankingsCalculator
from itertools import combinations

class Logistic(object):

    def __init__(self, reg_season, tourney):

        """
        Wrapper for sklearn Logistic Regression for Kaggle NCAA Tourney

        """
        self.Ranks = RankingsCalculator(reg_season, tourney)
        self.Logit = linear_model.LogisticRegression()
        self.team_features = []

    def extract_features(self):
        pass

    def train(self):
        X = self.build_matrix()
        scaled_X = preprocessing.scale(X[:, :-1])
        Y = X[:, -1]
        self.Logit.fit(scaled_X, Y)

    def predict(self, infile, season):
        """
        Infile is tourney_seeds.csv

        Output is predictions, index
        """
        team_list = []
        matrix = []
        index = []
        reader = csv.reader(open(infile, 'r'))
        for row in reader:
            if row[0] == season:
                team_list.append(row[2])
                ####MODIFY LATER FOR SEEDS
        team_list.sort()
        for (l_team, h_team) in combinations(team_list, 2):
            matrix.append(self.build_row(season, l_team, h_team))
            index.append((l_team, h_team))
        return self.Logit.predict_proba(preprocessing.scale(np.array(matrix))), index


    def build_row(self, season, l_team, h_team):
        temp_list = []
        l_dict = self.Ranks.reg_season[season][l_team]
        h_dict = self.Ranks.reg_season[season][h_team]
        temp_list.append(self.Ranks.calc_RPI(season, l_team))
        temp_list.append(l_dict['net_score'])
        temp_list.append(self.Ranks.calc_SOS(season, l_team))
        temp_list.append(self.Ranks.calc_RPI(season, h_team))
        temp_list.append(h_dict['net_score'])
        temp_list.append(self.Ranks.calc_SOS(season, h_team))
        return temp_list

    def build_matrix(self):
        output_list = []
        for season in self.Ranks.tourney:
            for (l_team, h_team) in self.Ranks.tourney[season]:
                temp_list = self.build_row(season, l_team, h_team)
                temp_list.append(self.Ranks.tourney[season][(l_team, h_team)])
                output_list.append(temp_list)

        return np.array(output_list)


############
'''
Think about seeding and progression through the bracket
'''
#############


if __name__ == '__main__':
    L = Logistic('../raw_data/regular_season_results.csv', '../raw_data/tourney_results.csv')
    L.train()
    print L.predict('../raw_data/tourney_seeds.csv', 'S')