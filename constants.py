# constants.py
# This file contains useful constants that are used throughout the project.

# Base API information
BASE_URL = 'api.riotgames.com'
API_KEY = 'RGAPI-4177431d-41b0-4f13-a79b-359cbda2d9aa'

# Dataset directory structure
DATASET_DIRECTORIES = [
    './dataset',
    './dataset/challenger_players',
    './dataset/challenger_games'
]

# All the url routing regions used by the riot api
URL_REGIONS = [
    'br1',
    'eun1',
    'euw1',
    'jp1',
    'kr',
    'la1',
    'la2',
    'na1',
    'oc1',
    'tr1',
    'ru'
]

# All the regional routing regions used by the riot api
REGIONS = {
    'br1': 'americas',
    'eun1': 'europe',
    'euw1': 'europe',
    'jp1': 'asia',
    'kr': 'asia',
    'la1': 'americas',
    'la2': 'americas',
    'na1': 'americas',
    'oc1': 'asia',
    'tr1': 'europe',
    'ru': 'europe'
}

# Challenger games dataset headers
GAME_HEADERS = [
    # Blue team top
    'champ1.1',
    'kills1.1',
    'assists1.1',
    'deaths1.1',
    'gold1.1',

    # Blue team jungle
    'champ1.2',
    'kills1.2',
    'assists1.2',
    'deaths1.2',
    'gold1.2',

    # Blue team mid
    'champ1.3',
    'kills1.3',
    'assists1.3',
    'deaths1.3',
    'gold1.3',

    # Blue team bot
    'champ1.4',
    'kills1.4',
    'assists1.4',
    'deaths1.4',
    'gold1.4',

    # Blue team supp
    'champ1.5',
    'kills1.5',
    'assists1.5',
    'deaths1.5',
    'gold1.5',

    # Red team top
    'champ2.1',
    'kills2.1',
    'assists2.1',
    'deaths2.1',
    'gold2.1',

    # Red team jungle
    'champ2.2',
    'kills2.2',
    'assists2.2',
    'deaths2.2',
    'gold2.2',

    # Red team mid
    'champ2.3',
    'kills2.3',
    'assists2.3',
    'deaths2.3',
    'gold2.3',

    # Red team bot
    'champ2.4',
    'kills2.4',
    'assists2.4',
    'deaths2.4',
    'gold2.4',

    # Red team supp
    'champ1.5',
    'kills1.5',
    'assists1.5',
    'deaths1.5',
    'gold1.5',

    # Objectives
    'first blood',
    'first tower',
    'blue towers',
    'red towers',
    'blue dragons',
    'red dragons',
    'blue heralds',
    'red heralds',
    'blue barons',
    'red barons',
    
    # Target
    'win'
]