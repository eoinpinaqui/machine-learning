# scrape.py
# This file contains many useful functions for gathering data from the Riot API.

import requests
import constants
import json
from riotwatcher import LolWatcher, ApiError

watcher = LolWatcher(constants.API_KEY)


# Get data about a summoner
def get_summoner(summoner_name, region=constants.DEFAULT_REGION, api_key=constants.API_KEY):
    return LolWatcher(api_key).summoner.by_name(region, summoner_name)


# Get a list of matches from a summoner puuid
def get_match_list_for_summoner(summoner_puuid, region=constants.DEFAULT_REGION, api_key=constants.API_KEY):
    return LolWatcher(api_key).match.matchlist_by_puuid(region, summoner_puuid)


# Get a match by its id
def get_match_by_id(match_id, region=constants.DEFAULT_REGION, api_key=constants.API_KEY):
    return LolWatcher(api_key).match.by_id(region, match_id)


pin = get_summoner('pinaquack')
matches = get_match_list_for_summoner(pin['puuid'], region='europe')
match = get_match_by_id(matches[0], region='europe')

print(json.dumps(match, sort_keys=False, indent=4))
