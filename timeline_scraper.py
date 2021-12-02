# scrape.py
# This file contains many useful functions for gathering data from the Riot API.

import pandas as pd
from tqdm import tqdm
import csv
import os
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
    csv_file = open('./dataset/challenger_players_timeline/challenger_players_timeline.' + region + '.csv', 'w',
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
    create_dataset_structure()
    get_challenger_players(region)
    players = pd.read_csv(
        './dataset/challenger_players_timeline/challenger_players_timeline.' + region + '.csv').values.tolist()

    # Loop through each player and get their most recent n games
    games_5 = {}
    games_10 = {}
    games_15 = {}
    games_20 = {}
    games_full = {}

    for i in tqdm(range(len(players))):
        player = players[i]
        try:
            p = watcher.summoner.by_name(region, player[1])
            games = watcher.match.matchlist_by_puuid(constants.REGIONS[region], p['puuid'], count=n, type='ranked')
            for game in games:
                data = watcher.match.timeline_by_match(constants.REGIONS[region], game)
                games_5[game] = get_game_info(data, 6)
                games_10[game] = get_game_info(data, 11)
                games_15[game] = get_game_info(data, 16)
                games_20[game] = get_game_info(data, 21)
                games_full[game] = get_game_info(data, 120)
        except:
            print("Couldn't find " + player[1])

    # Open a new csv file to write the data to
    meta_file = open('./dataset/challenger_games_timeline_2/challenger_games_timeline_metadata' + '.' + region + '.csv', 'w',
                     encoding='utf-8', newline='')
    meta_writer = csv.writer(meta_file)
    meta_writer.writerow(constants.META_HEADERS)

    for game in games_full:
        meta_writer.writerow(games_full[game][0])
    meta_file.close()

    setCount = 1
    for x in [games_5, games_10, games_15, games_20, games_full]:

        csv_file = open('./dataset/challenger_games_timeline_2/challenger_games_timeline_' + str(setCount) + '.' + region + '.csv', 'w',
                        encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(constants.GAME_HEADERS_TIMELINE)

        for game in x:
            csv_writer.writerow(x[game][1])

        csv_file.close()
        setCount += 1


def get_game_info(data, max_frames):
    meta = []
    metaData = data["metadata"]
    meta.append(metaData["dataVersion"])
    meta.append(metaData["matchId"])
    meta.append(metaData["participants"])

    frames = data["info"]["frames"]
    gameData = []
    totalCs, totalXP, kills, assists, deaths = (([0] * 10) for i in range(5))
    first_blood, first_tower, winning_team, bDragon, rDragon, bRift, rRift, bTowers, rTowers, bInhib, rInhib = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i, frame in enumerate(frames):
        if i < max_frames:
            events = frame["events"]
            for event in events:
                if event["type"] == "CHAMPION_SPECIAL_KILL":
                    if event["killType"] == "KILL_FIRST_BLOOD":
                        first_blood = event["killerId"]

                if event["type"] == "CHAMPION_KILL":
                    kills[int(event["killerId"] - 1)] += 1
                    deaths[int(event["victimId"] - 1)] += 1
                    if "assistingParticipantIds" in event:
                        for assist in event["assistingParticipantIds"]:
                            assists[assist - 1] += 1

                if event["type"] == "ELITE_MONSTER_KILL":
                    if event["monsterType"] == "DRAGON":
                        if event["killerTeamId"] == 100:
                            bDragon += 1
                        if event["killerTeamId"] == 200:
                            rDragon += 1

                    if event["monsterType"] == "RIFTHERALD":
                        if event["killerTeamId"] == 100:
                            bRift += 1
                        if event["killerTeamId"] == 200:
                            rRift += 1

                if event["type"] == "BUILDING_KILL":
                    if event["buildingType"] == "TOWER_BUILDING":
                        if event["teamId"] == 100:
                            if rTowers + bTowers == 0:
                                first_tower = event["killerId"]
                            rTowers += 1
                        if event["teamId"] == 200:
                            if rTowers + bTowers == 0:
                                first_tower = event["killerId"]
                            bTowers += 1

                    if event["buildingType"] == "INHIBITOR_BUILDING":
                        if event["teamId"] == 100:
                            rInhib += 1
                        if event["teamId"] == 200:
                            bInhib += 1

            if i == max_frames - 1 or i == len(frames) - 1:
                for x in frame["participantFrames"]:
                    totalCs[int(x) - 1] = frame["participantFrames"][x]["jungleMinionsKilled"] + frame["participantFrames"][x]["minionsKilled"]
                    totalXP[int(x) - 1] = frame["participantFrames"][x]["level"]

    endGame = frames[-1]["events"]
    for event in endGame:
        if event["type"] == "GAME_END":
            if event["winningTeam"] == 100:
                winning_team = 1
            if event["winningTeam"] == 200:
                winning_team = -1
            game_length = event["timestamp"]

    gameData.append(first_blood)
    gameData.append(first_tower)
    gameData.append(bDragon)
    gameData.append(rDragon)
    gameData.append(bTowers)
    gameData.append(rTowers)
    gameData.append(bRift)
    gameData.append(rRift)
    gameData.append(bInhib)
    gameData.append(rInhib)

    for i in range(10):
        gameData.append(totalCs[i])
        gameData.append(totalXP[i])
        gameData.append(kills[i])
        gameData.append(deaths[i])
        gameData.append(assists[i])

    meta.append(game_length)
    gameData.append(winning_team)
    return [meta, gameData]


get_challenger_games('na1', 50)
get_challenger_games('euw1', 50)
