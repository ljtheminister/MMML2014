import pandas as pd
import numpy as np
from numpy import ndarray
from collections import defaultdict

# read data
reg_results = pd.read_csv('regular_season_results.csv', header=0)
tourney_results = pd.read_csv('tourney_results.csv', header=0)
seasons = pd.read_csv('seasons.csv', header=0)
teams = pd.read_csv('teams.csv', header=0)
seeds = pd.read_csv('tourney_seeds.csv', header=0)
slots = pd.read_csv('tourney_slots.csv', header=0)


# dictionary for win-loss percentage
data = dict()
for season in seasons['season']:
    data[season] = dict()

for row_idx in xrange(len(reg_results)):
    game = reg_results.ix[row_idx, :]
    season = game['season']
    wteam = game['wteam']
    lteam = game['lteam']

    wins = data[season][wteam].get('W', 0)
    data[season][wteam]['W'] = wins + 1
    losses = data[season][lteam].get('L', 0)
    data[season][lteam]['L'] = losses + 1

    wteam_opps = data[season][wteam].get('opponents', [])
    lteam_opps = data[season][lteam].get('opponents', [])
    wteam_opps.append(lteam)
    lteam_opps.append(wteam)

    data[season][wteam]['opponents'] = wteam_opps
    data[season][lteam]['opponents'] = lteam_opps

    # calculate SOS for each season

SOS = dict()
for season in seasons['season']:
    SOS[season] = dict()

for row_idx in xrange(len(reg_results)):
    game = reg_results.ix[row_idx, :]
    season = game['season']
    wteam = game['wteam']
    lteam = game['lteam']










