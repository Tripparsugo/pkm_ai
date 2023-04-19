from typing import Optional, Union

from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder
from poke_env.environment import Pokemon
from poke_env.environment.move import Move

import vectorization
from utils import get_model


class DeepPlayer(Player):

    def __init__(self, player_configuration: Optional[PlayerConfiguration] = None, *, avatar: Optional[int] = None,
                 battle_format: str = "gen8randombattle", log_level: Optional[int] = None,
                 max_concurrent_battles: int = 1, save_replays: Union[bool, str] = False,
                 server_configuration: Optional[ServerConfiguration] = None, start_timer_on_battle_start: bool = False,
                 start_listening: bool = True, ping_interval: Optional[float] = 20.0,
                 ping_timeout: Optional[float] = 20.0, team: Optional[Union[str, Teambuilder]] = None,
                 verbose=False) -> None:
        super().__init__(player_configuration, avatar=avatar, battle_format=battle_format, log_level=log_level,
                         max_concurrent_battles=max_concurrent_battles, save_replays=save_replays,
                         server_configuration=server_configuration,
                         start_timer_on_battle_start=start_timer_on_battle_start, start_listening=start_listening,
                         ping_interval=ping_interval, ping_timeout=ping_timeout, team=team)
        self.model = get_model()
        self.verbose = verbose

    def choose_move(self, battle):
        available_actions = battle.available_moves + battle.available_switches
        if len(available_actions) == 0:
            return self.choose_random_move(battle)

        possible_orders = []
        for aa in available_actions:
            order = self.create_order(aa)
            v = vectorization.vectorizeTurnInfo(battle, order, True)
            reward = self.model.predict([v])[0][0]
            possible_orders.append(dict(order=order, reward=reward))

        best_order = max(possible_orders, key=lambda po: po["reward"])["order"]

        if (self.verbose):
            for po in possible_orders:
                o = po["order"].order
                r = po["reward"]

                if isinstance(o, (Pokemon,)):
                    print(f"SWAP to {o.species}   ", end='')

                if isinstance(o, (Move,)):
                    print(f"USE  {o.id}   ", end='')

                print(f"R: {r}")

            return best_order
