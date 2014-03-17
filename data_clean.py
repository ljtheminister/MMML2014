import pandas as pd
import numpy as np

# read data
reg_results = pd.read_csv('regular_season_results.csv', header=0)
tourney_results = pd.read_csv('tourney_results.csv', header=0)
seasons = pd.read_csv('seasons.csv', header=0)
teams = pd.read_csv('teams.csv', header=0)
seeds = pd.read_csv('seeds.csv', header=0)
slots = pd.read_csv('slots.csv', header=0)


