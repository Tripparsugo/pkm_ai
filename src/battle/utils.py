from typing import List, Optional, TypeVar
from poke_env.environment import PokemonType, Gen8Pokemon
from poke_env.environment.move import Move
import numpy as np

DEFENSE_ABILITIES_TO_IMMUNITIES = {
    "levitate": [PokemonType.GROUND],
    "voltabsorb": [PokemonType.ELECTRIC],
    "motordrive": [PokemonType.ELECTRIC],
    "waterabsorb": [PokemonType.WATER],
    "stormdrain": [PokemonType.WATER],
    "flashfire": [PokemonType.FIRE],
    "sapsipper": [PokemonType.GRASS],
}


def compute_damage_multiplier(attack_type: PokemonType, defense_types: List[PokemonType],
                              defense_ability: Optional[str] = None):
    if defense_ability and defense_ability in DEFENSE_ABILITIES_TO_IMMUNITIES:
        for dt in defense_types:
            if dt in DEFENSE_ABILITIES_TO_IMMUNITIES[defense_ability]:
                return 0

    dt1_v = defense_types[0]
    dt2_v = defense_types[1] if len(defense_types) > 1 else None

    return attack_type.damage_multiplier(dt1_v, dt2_v)


def compute_move_effective_power(attacking_pokemon: Gen8Pokemon,
                                 defending_pokemon: Gen8Pokemon, move: Move, ignore_defending: bool = False) -> float:
    attacking_pokemon_types = attacking_pokemon.types

    move_base_power = move.base_power
    move_type = move.type

    is_stab = move_type in attacking_pokemon_types
    stab_mul = 1.5 if is_stab else 1

    if ignore_defending:
        total_pow = move_base_power * stab_mul
        return total_pow

    defending_pokemon_types = [x for x in defending_pokemon.types]

    defending_ability = defending_pokemon.ability
    type_mul = compute_damage_multiplier(move_type, defending_pokemon_types, defending_ability)
    total_pow = move_base_power * stab_mul * type_mul
    return total_pow


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


T = TypeVar("T")


def one_hot_encode(categories: [T], values: [T]):
    categories = sorted(categories)
    oh = []
    for c in categories:
        oh.append(1. if c in values else 0.)
    return oh
# only difference
