# scrape.py
# This file contains many useful functions for gathering data from the Riot API.

import pandas as pd
from tqdm import tqdm
import csv
import os
import json

from riotwatcher import LolWatcher

import constants

# Create the league watcher object (handles rate limiting when scraping data)
watcher = LolWatcher(constants.API_KEY)


# Function to create the directory structure to hold the data set
def create_dataset_structure():
    for directory in constants.DATASET_DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Function to get all the challenger players from the riot api
def get_challenger_players(region):
    # Get the list of challenger players from teh riot api
    challengers = watcher.league.challenger_by_queue(region, 'RANKED_SOLO_5x5')

    # Open a new csv file to write the data to
    csv_file = open('./dataset/challenger_players/challenger_players.' + region + '.csv', 'w',
                    encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)

    # Write the data to the open csv
    write_headers = True
    for player in challengers['entries']:
        if write_headers:
            csv_writer.writerow(player.keys())
            write_headers = False
        csv_writer.writerow(player.values())
    csv_file.close()


# Function to get challenger games from the riot api
# considers most recent n games of the challenger players in a region
def get_challenger_games(region, n):
    # Get all of the challenger players for the given region
    # create_dataset_structure()
    # get_challenger_players(region)
    players = pd.read_csv('./dataset/challenger_players/challenger_players.' + region + '.csv').values.tolist()

    # Loop through each player and get their most recent n games
    region_games = {}
    for i in tqdm(range(len(players))):
        player = players[i]
        try:
            p = watcher.summoner.by_name(region, player[1])
            games = watcher.match.matchlist_by_puuid(constants.REGIONS[region], p['puuid'], count=n)
            for j in tqdm(range(len(games))):
                game = games[j]
                region_games[game] = get_game_info(constants.REGIONS[region], game)
        except:
            player = ''

    # Open a new csv file to write the data to
    csv_file = open('./dataset/challenger_games/challenger_games_big.' + region + '.csv', 'w',
                    encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)

    # Write the data to the open csv
    csv_writer.writerow(constants.GAME_HEADERS)
    for game in region_games:
        csv_writer.writerow(region_games[game])

    csv_file.close()


def get_game_info(region, game_id):
    game = watcher.match.by_id(region, game_id)['info']
    result = []
    for x in game['participants']:
        result.append(x['championId'])
        result.append(x['kills'])
        result.append(x['assists'])
        result.append(x['deaths'])
        result.append(x['goldEarned'])

    blue_team = game['teams'][0]
    red_team = game['teams'][1]

    result.append(1 * blue_team['objectives']['champion']['first'] + -1 * red_team['objectives']['champion']['first'])
    result.append(1 * blue_team['objectives']['tower']['first'] + -1 * red_team['objectives']['tower']['first'])
    result.append(blue_team['objectives']['tower']['kills'])
    result.append(red_team['objectives']['tower']['kills'])
    result.append(blue_team['objectives']['dragon']['kills'])
    result.append(red_team['objectives']['dragon']['kills'])
    result.append(blue_team['objectives']['riftHerald']['kills'])
    result.append(red_team['objectives']['riftHerald']['kills'])
    result.append(blue_team['objectives']['baron']['kills'])
    result.append(red_team['objectives']['baron']['kills'])

    result.append(1 * blue_team['win'] + -1 * red_team['win'])

    return result


# Gets the timeline for a game
p = watcher.summoner.by_name('euw1', 'pinaquack')
games = watcher.match.matchlist_by_puuid('europe', p['puuid'])
game = watcher.match.timeline_by_match('europe', games[0])

j = json.dumps(game, indent=4)
f = open('timeline.json', 'w')
f.write(j)
f.close()
