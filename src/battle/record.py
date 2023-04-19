import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from poke_env.environment import AbstractBattle
from poke_env.player import Player, BattleOrder
import math
from src.battle.encoders import Encoder
from src.battle.utils import compute_damage_multiplier


class RewardAssigner(ABC):
    @abstractmethod
    def state_values_to_rewards(self, sv: List[float]) -> List[float]:
        pass

    @abstractmethod
    def state_to_value(self, battle: AbstractBattle):
        pass


# TODO rm after resolving bug
other_battle = None


class ConcreteRewardAssigner(RewardAssigner):

    def __init__(self, window_size: int = 1, decay: float = 0.5, hp_weight=1,
                 faint_weight=0.3,
                 status_weight=0.2,
                 type_weight=0.1) -> None:
        super().__init__()
        self.window_size = window_size
        self.decay = decay
        self.faint_weight = faint_weight
        self.status_weight = status_weight
        self.type_weight = type_weight
        self.hp_weight = hp_weight

        assert self.window_size > 0

    @staticmethod
    def _weight_sum(window_size, decay):
        w = 1
        ws = 0
        for i in range(0, window_size):
            ws = ws + w
            w = w * decay

        return ws

    def state_values_to_rewards(self, sv: List[float]) -> List[float]:
        rs = []
        for i in range(0, len(sv) - 1):
            w = 1
            r = 0
            actual_window_size = min(self.window_size, len(sv) - i - 1)
            for j in range(i, i + actual_window_size):
                r = r + sv[j + 1] - sv[j]
                w = w * self.decay
            rs.append(r)
            # assign the reward to first action when the states available are smaller than the window
            iw = self._weight_sum(self.window_size, self.decay) - self._weight_sum(actual_window_size, self.decay)
            r_adj = (sv[i + 1] - sv[i]) * iw
            rs[i] = rs[i] + r_adj

        return rs

    def state_to_value(self, battle: AbstractBattle) -> float:
        opp_hp = 6
        oppo_status = 0
        oppo_faint = 0
        for p in battle.opponent_team.values():
            opp_hp = opp_hp - (1 - p.current_hp_fraction)
            if p.status is not None and p.status.name != "FNT":
                oppo_status = oppo_status + 1

            if p.status is not None and p.status.name == "FNT":
                oppo_faint = oppo_faint + 1

        own_hp = 0
        own_status = 0
        own_faint = 0
        for p in battle.team.values():
            # own hp has more precision
            own_hp = own_hp + math.ceil(p.current_hp_fraction * 100) / 100
            if p.status is not None and p.status.name != "FNT":
                own_status = own_status + 1

            if p.status is not None and p.status.name == "FNT":
                own_faint = own_faint + 1

        hp_value = own_hp - opp_hp
        hp_value = hp_value * self.hp_weight

        faint_value = oppo_faint - own_faint
        faint_value = faint_value * self.faint_weight

        status_value = oppo_status - own_status
        status_value = status_value * self.status_weight

        offense_type_value = 0
        opponent_offense_type_value = 0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            own_types = battle.active_pokemon.types
            opponent_types = battle.opponent_active_pokemon.types
            opponent_ability = battle.opponent_active_pokemon.ability
            own_ability = battle.active_pokemon.ability
            offense_type_value = max(
                compute_damage_multiplier(own_types[0], opponent_types, opponent_ability),
                -1 if own_types[1] is None else
                compute_damage_multiplier(own_types[1], opponent_types, opponent_ability)
            )
            opponent_offense_type_value = max(
                compute_damage_multiplier(opponent_types[0], own_types, own_ability),
                -1 if opponent_types[1] is None else
                compute_damage_multiplier(opponent_types[1], own_types, own_ability)
            )

        type_value = offense_type_value - opponent_offense_type_value
        type_value = type_value * self.type_weight

        state_value = hp_value + faint_value + status_value + type_value

        global other_battle
        if other_battle != battle:
            other_battle = battle

        return state_value


@dataclass
class TurnRecord:
    encoding: List[float]
    reward: Optional[float]
    state_value: float
    order: Optional[BattleOrder]
    order_i: Optional[int]
    player_name: Optional[str]
    turn_n: Optional[int]


class BattleRecorder(ABC):
    def __init__(self, encoder: Encoder, ra: RewardAssigner) -> None:
        super().__init__()
        self.battle_to_turns: Dict[AbstractBattle, List[TurnRecord]] = dict()
        self.encoder = encoder
        self.ra = ra

    def push_turn(self, player: Player, battle: AbstractBattle, order: Optional[BattleOrder], order_i: int):
        if battle not in self.battle_to_turns:
            self.battle_to_turns[battle] = []

        encoding = self.encoder.encode_battle(battle)
        state_value = self.ra.state_to_value(battle)
        self.battle_to_turns[battle].append(
            TurnRecord(encoding=encoding, state_value=state_value, reward=None,
                       order=order, order_i=order_i,
                       player_name=player._username,
                       turn_n=len(self.battle_to_turns[battle])

                       )
        )
        # assign rewards once battle is finished
        if battle.finished:
            trs = self.battle_to_turns[battle]
            state_values = [tr.state_value for tr in trs]
            rs = self.ra.state_values_to_rewards(state_values)
            for i, r in enumerate(rs):
                trs[i].reward = r

    def reset(self):
        self.battle_to_turns = dict()
