__author__ = 'Nakis'
import pandas as pd
import csv
from collections import defaultdict
import numpy as np


class RankingsCalculator(object):

    def __init__(self, reg_season, tourney, seeds):

        self.reg_season = defaultdict(dict)
        self.tourney = defaultdict(dict)
        self.tourney_seeds = defaultdict(dict)

        self._process_regular(reg_season)
        self._process_tourney(tourney)
        self._process_seeds(seeds)

    def _process_regular(self, filename):
        with open(filename, 'r') as f:
            team_stuff = csv.reader(f)
            team_stuff.next()
            for row in team_stuff:
                season = row[0]
                wteam = row[2]
                lteam = row[4]
                diff = int(row[3]) - int(row[5])
                win_team_dict = self.reg_season[season].get(wteam,
                                                       {'wins': 0,
                                                        'home_wins': 0,
                                                        'neutral_wins': 0,
                                                        'losses': 0,
                                                        'home_losses': 0,
                                                        'neutral_losses': 0,
                                                        'opponents': [],
                                                        'net_score': 0}
                                                        )

                lose_team_dict = self.reg_season[season].get(lteam,
                                                       {'wins': 0,
                                                        'home_wins': 0,
                                                        'neutral_wins': 0,
                                                        'losses': 0,
                                                        'home_losses': 0,
                                                        'neutral_losses': 0,
                                                        'opponents': [],
                                                        'net_score': 0}
                                                        )
                win_team_dict['wins'] += 1
                lose_team_dict['losses'] += 1
                win_team_dict['opponents'].append((lteam, 'W'))
                lose_team_dict['opponents'].append((wteam, 'L'))
                win_team_dict['net_score'] += diff
                lose_team_dict['net_score'] -= diff
                if row[6] == 'H':
                    win_team_dict['home_wins'] += 1
                elif row[6] == 'A':
                    lose_team_dict['home_losses'] += 1
                else:
                    win_team_dict['neutral_wins'] += 1
                    lose_team_dict['neutral_losses'] += 1
                self.reg_season[season][wteam] = win_team_dict
                self.reg_season[season][lteam] = lose_team_dict

    def _process_tourney(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for row in reader:
                season = row[0]
                wteam = row[2]
                lteam = row[4]
                if wteam < lteam:
                    self.tourney[season][(wteam, lteam)] = 1
                else:
                    self.tourney[season][(lteam, wteam)] = 0

    def _process_seeds(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for row in reader:
                season = row[0]
                seed = row[1][-2:]
                region = row[1][0]
                team = row[2]
                self.tourney_seeds[season][team] = []
                #(1,15) -> n-1
                for i in [str(x) for x in range(1,15)]:
                    if i != seed:
                        self.tourney_seeds[season][team].append(0)
                    else:
                        self.tourney_seeds[season][team].append(1)
                #again, n-1
                for j in ['W', 'X', 'Y']:
                    if j != season:
                        self.tourney_seeds[season][team].append(0)
                    else:
                        self.tourney_seeds[season][team].append(1)




    def calc_RPI(self, season, team):
        WP = self.calc_WP(season, team)
        OWP = self.calc_OWP(season, team)
        OOWP = self.calc_OOWP(season, team)
        return .25 * WP + .5 * OWP + .25 * OOWP

    def calc_SOS(self, season, team):
        OWP = self.calc_OWP(season, team)
        OOWP = self.calc_OOWP(season, team)
        return (2 * OWP + OOWP) / 3.0

    def calc_WP(self, season, team):
        team_dict = self.reg_season[season][team]
        num = team_dict['neutral_wins'] + 0.6 * team_dict['home_wins'] + 1.4 * (team_dict['wins']
                                                                                   - team_dict['home_wins']
                                                                                   - team_dict['neutral_wins'])
        denom = team_dict['neutral_losses'] + 1.4 * team_dict['home_losses'] + 0.6 * (team_dict['losses']
                                                                                   - team_dict['home_losses']
                                                                                   - team_dict['neutral_losses'])
        return num / (num + denom)

    def calc_OWP(self, season, team):
        season_dict = self.reg_season[season]
        team_dict = season_dict[team]
        owp = 0
        for opponent in team_dict['opponents']:
            op_wins = season_dict[opponent[0]]['wins']
            op_losses = season_dict[opponent[0]]['losses']
            op_times_met = 0
            if opponent[1] == 'W':
                op_losses -= 1
            else:
                op_wins -= 1
            owp += (op_wins/float(op_wins + op_losses))
        return owp / float(len(team_dict['opponents']))

    def calc_OOWP(self, season, team):
        season_dict = self.reg_season[season]
        team_dict = season_dict[team]
        oopw = 0
        for opponent in team_dict['opponents']:
            o_team = opponent[0]
            o_score = 0
            kounter = 0
            for o_opponent in season_dict[o_team]['opponents']:
                o_score += (season_dict[o_opponent[0]]['wins'] / float(season_dict[o_opponent[0]]['wins'] + season_dict[o_opponent[0]]['losses']))
                kounter += 1
            oopw += (o_score / float(kounter))
        return oopw / (float(len(team_dict['opponents'])))




if __name__ == '__main__':

    '''
    file1 = '/Users/Nakis/PycharmProjects/Kaggle/NCAA/raw_data/regular_season_results.csv'
    file2 = '/Users/Nakis/PycharmProjects/Kaggle/NCAA/raw_data/tourney_results.csv'
    RC = RankingsCalculator(file1)
    sos_list = []
    rpi_list = []
    for i in range(502, 855):
        try:
            sos_list.append((i, RC.calc_SOS('A', str(i))))
            rpi_list.append((i, RC.calc_RPI('A', str(i))))
        except KeyError:
            pass
    sos_list.sort(key=lambda x:x[1], reverse = True)
    rpi_list.sort(key=lambda x:x[1], reverse = True)
    for guy in rpi_list[:15]:
        print guy[0], rpi_list.index(guy)
    '''