from typing import Optional, TypeVar

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import BattleOrder
from poke_env.environment import Pokemon
from poke_env.environment.move import Move
from poke_env.environment import PokemonType

POKEMON_TYPES = [
    "normal",
    "fire",
    "water",
    "grass",
    "electric",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dark",
    "dragon",
    "steel",
    "fairy"
]

POKEMON_STATUSES = [
    "fnt",
    "tox",
    "slp",
    "par",
    "brn",
    "frz",
    "psn"
]

POKEMON_ABILITIES = [
    "intimidate",
    "wonderguard",
    "speedboost",
    "protean",
    "multiscale",
    "levitate",
    "magicguard",
    "prankster"
]

MOVE_CATEGORIES = [
    "physical",
    "special",
    "status"
]

ITEMS = [
    "choicescarf",
    "choiceband",
    "leftovers",
    "lifeorb",
    "baloon",
    "heavydutyboots",
    "focussash"
]

WEATHER_STATUSES = [
    "sandstorm",
    "sunnyday",
    "raindance"
]

UNIQUE_MOVES = [
    "stealthrock",
    "rapidspin",
    "defog",
    "roost",
    "rest",
    "toxicspikes",
    "toxic",
    "thunderwave"
]

FIELD_CONDITIONS = [
    "stealthrock"
]

BOOST_TARGETS = ["atk", "spa", "def", "def", "spd", "spe"]
STATS = ["hp"] + BOOST_TARGETS

T = TypeVar("T")


def oneHotEncode(categories: [T], values: [T]):
    categories = sorted(categories)
    oh = []
    for c in categories:
        oh.append(1. if c in values else 0.)
    return oh


def vectorize_player_move(move: Optional[Move], battle_order: Optional[BattleOrder], valid):
    if not valid:
        size = 6 + len(vectorize_dex_move(None, False))
        return [0.] * size

    isSelectedMove_encoding = [1.] if battle_order.order == move else [0.]
    pp_encoding = [move.current_pp / 40.]
    maxpp_encoding = [move.max_pp / 40.]
    disabled_encoding = [0.]
    used_encoding = [0.]
    valid_encoding = [1.] if valid else [0.]
    dex_endoding = vectorize_dex_move(move, valid)
    encoding = isSelectedMove_encoding + valid_encoding + pp_encoding \
               + maxpp_encoding + disabled_encoding + used_encoding + dex_endoding
    return encoding


def vectorize_dex_move(dexMove: Optional[Move], valid: bool):
    if not valid:
        size = 8 + 5 + len(BOOST_TARGETS) + len(POKEMON_TYPES) + len(MOVE_CATEGORIES) + len(UNIQUE_MOVES)
        return [0.] * size

    accuracy_encoding = [1.] if dexMove.accuracy else [dexMove.accuracy]
    basePower_encoding = [dexMove.base_power / 250.]
    priority = dexMove.priority
    priorityEncoding = [0., 0., 0., 0., 0.]

    if priority < 0:
        priorityEncoding[0] = 1.
    elif priority < len(priorityEncoding):
        priorityEncoding[priority] = 1.
    else:
        priorityEncoding[-1] = 1.

    type = dexMove.type
    typeEncoding = oneHotEncode(POKEMON_TYPES, [type.name.lower()])
    category = dexMove.category.name.lower()  # TODO
    categoryEncoding = oneHotEncode(MOVE_CATEGORIES, [category])
    useTargetOffensive_encoding = [1.] if dexMove.use_target_offensive else [0.]
    ignoreImmunity_encoding = [1.] if dexMove.ignore_immunity else [0.]
    ignoreDefensive_encoding = [1.] if dexMove.ignore_defensive else [0.]
    volatileStatus_encoding = [1.] if dexMove.volatile_status else [0.]  # protect

    uniqueEncoding = oneHotEncode(UNIQUE_MOVES, [dexMove.id])

    boosts = dexMove.boosts if dexMove.boosts else dict()  # {atk:, spa:,def :,spd:, spe:}
    boostEncoding = []

    for t in BOOST_TARGETS:
        boost_value = boosts[t] / 3 if t in boosts else 0
        boostEncoding.append(boost_value)
    target_encoding = [1.] if dexMove.target == "allySide" else [0.]  # allySide | opponentSide
    valid_encoding = [1.] if valid else [0.]

    encoding = valid_encoding + accuracy_encoding + basePower_encoding \
               + uniqueEncoding + boostEncoding + priorityEncoding + typeEncoding + categoryEncoding + useTargetOffensive_encoding \
               + ignoreDefensive_encoding + ignoreImmunity_encoding + volatileStatus_encoding + target_encoding

    return encoding


def vectorizePlayerPokemon(pokemon: Optional[Pokemon], battleOrder: Optional[BattleOrder], valid: bool) -> [float]:
    if not valid:
        size = len(vectorizePlayerPokemonShort(None, None, False)) \
               + len(vectorize_player_move(None, None, False)) * 4

        return [0] * size

    shortEncoding = vectorizePlayerPokemonShort(pokemon, battleOrder, valid)
    movesEncoding = []
    move_keys = list(pokemon.moves.keys())
    for i in range(0, 4):
        if i >= len(pokemon.moves):
            movesEncoding += vectorize_player_move(None, battleOrder, False)
            continue

        move = pokemon.moves[move_keys[i]]
        movesEncoding += vectorize_player_move(move, battleOrder, True)

    encoding = shortEncoding + movesEncoding

    return encoding


def vectorizePlayerPokemonShort(pokemon: Optional[Pokemon], battleOrder: Optional[BattleOrder], valid: bool) -> [float]:
    if not valid:
        size = 5 + len(POKEMON_STATUSES) + len(POKEMON_ABILITIES) + len(ITEMS) + len(POKEMON_TYPES)
        return [0.] * size

    isSelectedSwap_encoding = [1.] if battleOrder.order == pokemon else [0.]
    abilityEncoding = oneHotEncode(POKEMON_ABILITIES, [pokemon.ability])
    itemEncoding = oneHotEncode(ITEMS, [pokemon.item])
    types_s = [t.name.lower() for t in pokemon.types if t is not None]
    typesEncoding = oneHotEncode(POKEMON_TYPES, types_s)
    active_encoding = [1.] if pokemon.active else [0.]
    boostEncoding = []
    for t in BOOST_TARGETS:
        boost_value = pokemon.boosts[t] / 3
        boostEncoding.append(boost_value)
    statEncoding = []
    for s in STATS:
        statEncoding.append(pokemon.base_stats[s] / 255)

    statusEncoding = oneHotEncode(POKEMON_STATUSES, [pokemon.status])
    hp_encoding = [pokemon.current_hp / 714]
    maxhp_encoding = [pokemon.max_hp / 714]
    valid_encoding = [1.] if valid else [0.]

    encoding = valid_encoding + isSelectedSwap_encoding + active_encoding + hp_encoding + maxhp_encoding \
               + statusEncoding + abilityEncoding + itemEncoding + typesEncoding

    return encoding


def vectorizeOpponentPokemonShort(pokemon: Optional[Pokemon], valid: bool) -> [float]:
    if not valid:
        size = 2 + len(POKEMON_TYPES) + len(STATS) + len(POKEMON_STATUSES)
        return [0.] * size

    statEncoding = []
    for s in STATS:
        statEncoding.append(pokemon.base_stats[s] / 255)

    types_s = [t.name.lower() for t in pokemon.types if t is not None]
    typesEncoding = oneHotEncode(POKEMON_TYPES, types_s)
    hp_fraction_encoding = [pokemon.current_hp_fraction]
    statusEncoding = oneHotEncode(POKEMON_STATUSES, [pokemon.status])
    active_encoding = [1.] if pokemon.active else [0.]
    encoding = statusEncoding + statEncoding + typesEncoding + active_encoding + hp_fraction_encoding
    return encoding


def vectorizeOpponentPokemon(pokemon: Optional[Pokemon], valid: bool) -> [float]:
    if not valid:
        size = len(vectorizeOpponentPokemonShort(pokemon, valid)) + len(vectorize_dex_move(None, False)) * 4
        return [0.] * size

    shortEnconding = vectorizeOpponentPokemonShort(pokemon, valid)
    moves_encoding = []
    move_keys = list(pokemon.moves.keys())

    for i in range(0, 4):
        if i >= len(pokemon.moves):
            vMove = vectorize_dex_move(None, False)
            moves_encoding += vMove
            continue
        move = pokemon.moves[move_keys[i]]
        vMove = vectorize_dex_move(move, True)
        moves_encoding += vMove

    encoding = shortEnconding + moves_encoding

    return encoding


def vectorizeTurnInfo(battle: Optional[AbstractBattle], playerAction: Optional[BattleOrder], valid: bool):
    if not valid:
        size = len(FIELD_CONDITIONS) * 2 + len(WEATHER_STATUSES) \
               + len(vectorizePlayerPokemon(None, None, False)) \
               + len(vectorizePlayerPokemonShort(None, None, False)) * 5 \
               + len(vectorizeOpponentPokemon(None, False)) \
               + len(vectorizeOpponentPokemonShort(None, False)) * 5
        return [0] * size
    weatherEncoding = oneHotEncode(WEATHER_STATUSES, battle.weather.keys())
    allyFieldEncoding = oneHotEncode(FIELD_CONDITIONS, battle.side_conditions.keys())
    enemyFieldEncoding = oneHotEncode(FIELD_CONDITIONS, battle.opponent_side_conditions.keys())
    encoding = weatherEncoding + allyFieldEncoding + enemyFieldEncoding
    # active pokemon must always be the first vectorized to preserve order of encoding
    active_pokemon = battle.active_pokemon
    team_keys = list(battle.team.keys())
    if active_pokemon is None:
        active_pokemon = battle.team[team_keys[0]]
    v = vectorizePlayerPokemon(active_pokemon, playerAction, True)
    encoding += v
    i = 0
    while i < 6:
        p = battle.team[team_keys[i]]
        if p == active_pokemon:
            i += 1
            continue
        v = vectorizePlayerPokemonShort(p, playerAction, True)
        encoding += v
        i += 1

    opponent_team_keys = list(battle.opponent_team.keys())
    opponent_active_pokemon = battle.opponent_active_pokemon
    if opponent_active_pokemon is None:
        opponent_active_pokemon = battle.opponent_team[opponent_team_keys[0]]
    v = vectorizeOpponentPokemon(opponent_active_pokemon, True)
    encoding += v
    i = 0
    while i < 6:
        if i >= len(battle.opponent_team):
            v = vectorizeOpponentPokemonShort(None, False)
            encoding += v
            i += 1
            continue

        p = battle.opponent_team[opponent_team_keys[i]]
        if p == opponent_active_pokemon:
            i += 1
            continue
        v = vectorizeOpponentPokemonShort(p, True)
        encoding += v
        i += 1
    return encoding
