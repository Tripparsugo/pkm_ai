from typing import Optional, Union

from poke_env import PlayerConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder
from src.battle.encoders import SimpleEncoder
import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.player.openai_api import ObservationType


class RLPEnvPlayer(Gen8EnvSinglePlayer):
    def __init__(self, opponent: Optional[Union[Player, str]],
                 player_configuration: Optional[PlayerConfiguration] = None, *, avatar: Optional[int] = None,
                 battle_format: Optional[str] = None, log_level: Optional[int] = None,
                 save_replays: Union[bool, str] = False, server_configuration: Optional[ServerConfiguration] = None,
                 start_listening: bool = True, start_timer_on_battle_start: bool = False,
                 ping_interval: Optional[float] = 20.0, ping_timeout: Optional[float] = 20.0,
                 team: Optional[Union[str, Teambuilder]] = None, start_challenging: bool = True,
                 use_old_gym_api: bool = True):
        super().__init__(opponent, player_configuration, avatar=avatar, battle_format=battle_format,
                         log_level=log_level, save_replays=save_replays, server_configuration=server_configuration,
                         start_listening=start_listening, start_timer_on_battle_start=start_timer_on_battle_start,
                         ping_interval=ping_interval, ping_timeout=ping_timeout, team=team,
                         start_challenging=start_challenging, use_old_gym_api=use_old_gym_api)

        self.encoder = SimpleEncoder()

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=5.0, status_value=0.5
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        encoding = SimpleEncoder().encode_battle(battle)
        return np.float32(encoding)

    def describe_embedding(self) -> Space:
        size = len(SimpleEncoder().encode_battle(None, False))

        low = np.zeros(size)
        high = np.ones(size)
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
