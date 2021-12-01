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

    for i in tqdm(range(len(players))):
        player = players[i]
        try:
            p = watcher.summoner.by_name(region, player[1])
            games = watcher.match.matchlist_by_puuid(constants.REGIONS[region], p['puuid'], count=n)
            for game in games:
                games_5[game] = get_game_info(constants.REGIONS[region], game, 6)
                games_10[game] = get_game_info(constants.REGIONS[region], game, 11)
                games_15[game] = get_game_info(constants.REGIONS[region], game, 16)
        except:
            print("Couldn't find " + player[1])

    # Open a new csv file to write the data to

    csv_file = open('./dataset/challenger_games_timeline/challenger_games_timeline_5.' + region + '.csv', 'w',
                    encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    meta_file = open('./dataset/challenger_games_timeline/challenger_games_timeline_metadata_5.' + region + '.csv', 'w',
                     encoding='utf-8', newline='')
    meta_writer = csv.writer(meta_file)

    meta_writer.writerow(constants.META_HEADERS)
    csv_writer.writerow(constants.GAME_HEADERS_TIMELINE)
    for game in games_5:
        meta_writer.writerow(games_5[game][0])
        csv_writer.writerow(games_5[game][1])

    csv_file.close()
    meta_file.close()

    csv_file = open('./dataset/challenger_games_timeline/challenger_games_timeline_10.' + region + '.csv', 'w',
                    encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    meta_file = open('./dataset/challenger_games_timeline/challenger_games_timeline_metadata_10.' + region + '.csv',
                     'w', encoding='utf-8', newline='')
    meta_writer = csv.writer(meta_file)

    meta_writer.writerow(constants.META_HEADERS)
    csv_writer.writerow(constants.GAME_HEADERS_TIMELINE)
    for game in games_10:
        meta_writer.writerow(games_10[game][0])
        csv_writer.writerow(games_10[game][1])

    csv_file.close()
    meta_file.close()

    csv_file = open('./dataset/challenger_games_timeline/challenger_games_timeline_15.' + region + '.csv', 'w',
                    encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    meta_file = open('./dataset/challenger_games_timeline/challenger_games_timeline_metadata_15.' + region + '.csv',
                     'w', encoding='utf-8', newline='')
    meta_writer = csv.writer(meta_file)

    meta_writer.writerow(constants.META_HEADERS)
    csv_writer.writerow(constants.GAME_HEADERS_TIMELINE)
    for game in games_15:
        meta_writer.writerow(games_15[game][0])
        csv_writer.writerow(games_15[game][1])

    csv_file.close()
    meta_file.close()


def get_game_info(region, game_id, max_frames):
    data = watcher.match.timeline_by_match(region, game_id)

    meta = []
    metaData = data["metadata"]
    meta.append(metaData["dataVersion"])
    meta.append(metaData["matchId"])
    meta.append(metaData["participants"])

    data = data["info"]["frames"]
    gameData = []
    totalCs, totalXP, kills, assists, deaths = (([0] * 10) for i in range(5))
    first_blood, first_tower, winning_team, bDragon, rDragon, bRift, rRift, bTowers, rTowers, bInhib, rInhib = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i, frames in enumerate(data):
        if i < max_frames:
            events = frames["events"]
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

            if i == max_frames - 1 or i == len(data) - 1:
                for x in frames["participantFrames"]:
                    totalCs[int(x) - 1] = frames["participantFrames"][x]["jungleMinionsKilled"] + \
                                          frames["participantFrames"][x]["minionsKilled"]
                    totalXP[int(x) - 1] = frames["participantFrames"][x]["xp"]

    endGame = data[-1]["events"]
    for event in endGame:
        if event["type"] == "GAME_END":
            winning_team = event["winningTeam"]

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

    gameData.append(winning_team)
    return [meta, gameData]


get_challenger_games('na1', 1)
