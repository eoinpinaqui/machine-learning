# scrape.py
# This file contains many useful functions for gathering data from the Riot API.

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
def get_challenger_players():
    # Loop through all of the regions
    for url_region in constants.URL_REGIONS:

        # Get the list of challenger players from teh riot api
        challengers = watcher.league.challenger_by_queue(url_region, 'RANKED_SOLO_5x5')

        # Open a new csv file to write the data to
        csv_file = open('./dataset/challenger_players/challenger_players.' + url_region + '.csv', 'w',
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


create_dataset_structure()
get_challenger_players()
