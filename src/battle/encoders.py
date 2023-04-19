from abc import ABC, abstractmethod
from typing import List, Optional
from poke_env.environment.move import Move

from poke_env.environment import AbstractBattle, Gen8Pokemon, Pokemon

from src.battle.utils import compute_move_effective_power, one_hot_encode

MOVE_CATEGORIES = [
    "physical",
    "special",
    "status"
]


# an encoder encodes the state of the battle into an array of floats. Used as input for NN.
class Encoder(ABC):
    @abstractmethod
    def encode_battle(self, battle: Optional[AbstractBattle], valid: bool = True) -> List[float]:
        pass

    @abstractmethod
    def encoding_size(self) -> int:
        pass


class SimpleEncoder(Encoder):
    def encode_battle(self, battle: Optional[AbstractBattle], valid: bool = True):
        if not valid:
            size = len(self._encode_pokemon(None, None, False)) * 12
            return [0] * size

        if not battle:
            raise Exception("must have battle if valid")

        team = []

        active_pokemon = battle.active_pokemon
        opponent_active_pokemon = battle.opponent_active_pokemon

        if active_pokemon:
            # active always first in encoding
            team.append(battle.active_pokemon)

        team = team + [x for x in battle.team.values() if not x.active]
        team_encoding = []
        for i in range(0, 6):
            p = team[i]
            e = self._encode_pokemon(p, opponent_active_pokemon)
            team_encoding = team_encoding + e

        opponent_team = []
        if opponent_active_pokemon:
            # active always first in encoding
            team.append(battle.opponent_active_pokemon)

        opponent_team = opponent_team + [x for x in battle.opponent_team.values() if not x.active]
        opponent_team_encoding = []
        for i in range(0, 6):
            if len(opponent_team) <= i:
                opponent_team_encoding = opponent_team_encoding + self._encode_pokemon(None, None, valid=False)
                continue

            p = opponent_team[i]
            e = self._encode_pokemon(p, active_pokemon)
            opponent_team_encoding = opponent_team_encoding + e

        return team_encoding + opponent_team_encoding

    def encoding_size(self):
        return len(self.encode_battle(None, False))

    def _encode_pokemon(self, pokemon, opponent_pokemon, valid=True):
        if not valid:
            size = 6 + 1 + len(self._encode_move(None, None, None, False)) * 4
            return [0] * size

        hp_fraction_encoding = [pokemon.current_hp_fraction]
        # TODO use actual for own
        stat_encoding = [x / 255 for x in pokemon.base_stats.values()]

        move_encodings = []
        moves = list(pokemon.moves.values())
        for i in range(0, 4):
            move = moves[i] if i < len(moves) else None
            move_encoding = self._encode_move(move, pokemon, opponent_pokemon, move is not None)
            move_encodings = move_encodings + move_encoding

        return hp_fraction_encoding + stat_encoding + move_encodings

    def _encode_move(self, move: Move, pokemon, opponent_pokemon, valid=True):
        if not valid:
            size = len(MOVE_CATEGORIES) + 2
            return [0] * size
        MAX_POW = 400
        power_encoding = [compute_move_effective_power(pokemon, opponent_pokemon,
                                                       move, ignore_defending=opponent_pokemon is None) / MAX_POW]
        category_encoding = one_hot_encode(MOVE_CATEGORIES, move.category.name.lower())

        is_boosting = move.boosts is not None
        boost_encoding = [1] if is_boosting else [0]
        return power_encoding + category_encoding + boost_encoding


class SimpleEncoderCP(Encoder):
    def encode_battle(self, battle: Optional[AbstractBattle], valid: bool = True):
        if not valid:
            size = len(self._encode_pokemon(None, None, False)) * 12
            return [0] * size

        if not battle:
            raise Exception("must have battle if valid")

        team = []

        active_pokemon = battle.active_pokemon
        opponent_active_pokemon = battle.opponent_active_pokemon

        if active_pokemon:
            # active always first in encoding
            team.append(battle.active_pokemon)

        team = team + [x for x in battle.team.values() if not x.active]
        team_encoding = []
        for i in range(0, 6):
            p = team[i]
            e = self._encode_pokemon(p, opponent_active_pokemon)
            team_encoding = team_encoding + e

        opponent_team = []
        if opponent_active_pokemon:
            # active always first in encoding
            team.append(battle.opponent_active_pokemon)

        opponent_team = opponent_team + [x for x in battle.opponent_team.values() if not x.active]
        opponent_team_encoding = []
        for i in range(0, 6):
            if len(opponent_team) <= i:
                opponent_team_encoding = opponent_team_encoding + self._encode_pokemon(None, None, valid=False)
                continue

            p = opponent_team[i]
            e = self._encode_pokemon(p, active_pokemon)
            opponent_team_encoding = opponent_team_encoding + e

        return team_encoding + opponent_team_encoding

    def encoding_size(self):
        return len(self.encode_battle(None, False))

    def _encode_pokemon(self, pokemon, opponent_pokemon, valid=True):
        if not valid:
            size = 6 + 1 + len(self._encode_move(None, None, None, False)) * 4
            return [0] * size

        hp_fraction_encoding = [pokemon.current_hp_fraction]
        # TODO use actual for own
        stat_encoding = [x / 255 for x in pokemon.base_stats.values()]

        move_encodings = []
        moves = list(pokemon.moves.values())
        for i in range(0, 4):
            move = moves[i] if i < len(moves) else None
            move_encoding = self._encode_move(move, pokemon, opponent_pokemon, move is not None)
            move_encodings = move_encodings + move_encoding

        return hp_fraction_encoding + stat_encoding + move_encodings

    def _encode_move(self, move: Move, pokemon, opponent_pokemon, valid=True):
        if not valid:
            size = len(MOVE_CATEGORIES) + 2
            return [0] * size
        MAX_POW = 400
        power_encoding = [compute_move_effective_power(pokemon, opponent_pokemon,
                                                       move, ignore_defending=opponent_pokemon is None) / MAX_POW]
        category_encoding = one_hot_encode(MOVE_CATEGORIES, move.category.name.lower())

        is_boosting = move.boosts is not None
        boost_encoding = [1] if is_boosting else [0]
        return power_encoding + category_encoding + boost_encoding


class SimplestEncoder(Encoder):
    def encode_battle(self, battle: Optional[AbstractBattle], valid: bool = True):
        if not valid:
            return len(self._encode_move(None, None, None, valid=False)) * 4

        encoding = []

        moves = list(battle.active_pokemon.moves.values())

        for i in range(0, 4):
            m = moves[i] if len(moves) > i else None
            if m is None:
                encoding = encoding + [0]
                continue

            e = self._encode_move(m, battle.active_pokemon, battle.opponent_active_pokemon, valid=True)
            encoding = encoding + e

        return encoding

    def _encode_move(self, move: Move, pokemon, opponent_pokemon, valid=True):
        if not valid:
            return [0] * 1
        MAX_POW = 400
        power_encoding = [compute_move_effective_power(pokemon, opponent_pokemon,
                                                       move, ignore_defending=opponent_pokemon is None) / MAX_POW]

        return power_encoding

    def encoding_size(self) -> int:
        return 4


# declare available encoders here
AVAILABLE_ENCODERS = {
    "simple": SimpleEncoder(),
    "simplest": SimplestEncoder()
}
