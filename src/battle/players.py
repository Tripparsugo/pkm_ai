from typing import Optional, Union, Awaitable, Callable

from poke_env import PlayerConfiguration, ServerConfiguration
from poke_env.environment import AbstractBattle, Battle, Pokemon
from poke_env.player import Player, BattleOrder, ForfeitBattleOrder
from poke_env.teambuilder import Teambuilder
from poke_env.environment import Battle

from src.battle.encoders import SimpleEncoder
from src.battle.evaluators import OrderEvaluator, PipelineEvaluator, PowerEvaluator, BoostOrderEvaluator, \
    BoostSwapEvaluator, BoostSwapToPowerful, DeepEvaluatorOut, make_evaluator, DeepEvaluator
from src.battle.model import load_latest_model_or_create_new, load_model, get_model_paths
from src.battle.pickers import Picker, BestPicker, SoftmaxPicker

callback_type = Callable[[Player, AbstractBattle, Optional[BattleOrder], Optional[int]], None]


# this player uses the given evaluator to assign values to the available action and then pass them to a picker.
# callbacks can be set to record turns.
class ConcretePlayer(Player):

    def __init__(self, evaluator: OrderEvaluator, picker: Picker, turn_callback: callback_type = None,
                 battle_end_callback: callback_type = None,
                 player_configuration: Optional[PlayerConfiguration] = None, *, avatar: Optional[int] = None,
                 battle_format: str = "gen8randombattle", log_level: Optional[int] = None,
                 max_concurrent_battles: int = 1, save_replays: Union[bool, str] = False,
                 server_configuration: Optional[ServerConfiguration] = None, start_timer_on_battle_start: bool = False,
                 start_listening: bool = True, ping_interval: Optional[float] = 20.0,
                 ping_timeout: Optional[float] = 20.0, team: Optional[Union[str, Teambuilder]] = None) -> None:
        super().__init__(player_configuration, avatar=avatar, battle_format=battle_format, log_level=log_level,
                         max_concurrent_battles=max_concurrent_battles, save_replays=save_replays,
                         server_configuration=server_configuration,
                         start_timer_on_battle_start=start_timer_on_battle_start, start_listening=start_listening,
                         ping_interval=ping_interval, ping_timeout=ping_timeout, team=team)
        self.evaluator = evaluator
        self.picker = picker
        self.turn_callback = turn_callback
        self.battle_end_callback = battle_end_callback

    def choose_move(self, battle: AbstractBattle) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        evaluations = self.evaluator.evaluate_orders(battle)
        non_random_evaluations = [x for x in evaluations if not x.is_random_order]
        if len(non_random_evaluations) == 0:
            # this seems to happen when the battle is over but for some reason it didn't register yet
            # when this happens we don't record the state and let the player handle it
            return self.choose_default_move()
        order_i = self.picker.pick_order(non_random_evaluations)
        ev = non_random_evaluations[order_i]

        if self.turn_callback:
            # callback to record battle, picked order and model order index
            self.turn_callback(self, battle, ev.battleOrder, ev.model_i)

        return ev.battleOrder

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super(ConcretePlayer, self)._battle_finished_callback(battle)
        if self.battle_end_callback is not None:
            self.battle_end_callback(self, battle, None, None)


def make_standard_player(server_configuration=None, player_configuration=None, use_soft=False) -> Player:
    evaluator = PipelineEvaluator([
        PowerEvaluator(),
        BoostOrderEvaluator(),
        BoostSwapEvaluator(),
        BoostSwapToPowerful()
    ])

    picker = SoftmaxPicker() if use_soft else BestPicker()

    player = ConcretePlayer(evaluator, picker, server_configuration=server_configuration,
                            player_configuration=player_configuration)
    return player


def make_latest_deep_player(server_configuration=None, player_configuration=None, use_soft=False, callback=None,
                            be_callback=None):
    evaluator = DeepEvaluatorOut()
    picker = SoftmaxPicker() if use_soft else BestPicker()

    player = ConcretePlayer(evaluator, picker, server_configuration=server_configuration,
                            player_configuration=player_configuration, turn_callback=callback,
                            battle_end_callback=be_callback)
    return player


def make_deep_player(model_path, server_configuration=None, player_configuration=None, use_soft=False, callback=None,
                     be_callback=None, encoder=SimpleEncoder()):
    model = load_model(model_path)
    evaluator = DeepEvaluator(model=model, encoder=encoder)
    picker = SoftmaxPicker() if use_soft else BestPicker()

    player = ConcretePlayer(evaluator, picker, server_configuration=server_configuration,
                            player_configuration=player_configuration, turn_callback=callback,
                            battle_end_callback=be_callback)
    return player


def make_player(config, server_configuration=None, player_configuration=None):
    evaluator = make_evaluator(config)
    picker = BestPicker()
    player = ConcretePlayer(evaluator, picker, server_configuration=server_configuration,
                            player_configuration=player_configuration)
    return player
