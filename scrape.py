# scrape.py
# This file contains many useful functions for gathering data from the Riot API.

import constants
import json
from riotwatcher import LolWatcher

# Create the league watcher object (handles rate limiting when scraping data)
watcher = LolWatcher(constants.API_KEY)

# Get information about a given summoner
summoner = watcher.summoner.by_name(constants.DEFAULT_URL_REGION, 'pinaquack')

# Get the match history for a given summoner
matches = watcher.match.matchlist_by_puuid(constants.DEFAULT_REGION, summoner['puuid'])

# Get the information for the first match in the list
match = watcher.match.by_id(constants.DEFAULT_REGION, matches[0])

with open('match.json', 'w', encoding='utf-8') as f:
    json.dump(match, f, sort_keys=False, ensure_ascii=False, indent=4)
