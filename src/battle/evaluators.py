import copy
from dataclasses import dataclass

from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.environment import AbstractBattle, Gen8Pokemon, Pokemon, Battle
from typing import List, Optional, Union, Awaitable, Dict
from abc import ABC, abstractmethod
from poke_env.environment.move import Move
import torch

from config import SELECTED_DEEP_ENCODER
from src.battle.encoders import SimpleEncoder, Encoder, AVAILABLE_ENCODERS
from src.battle.model import model_pick_to_order, load_latest_model_or_create_new
from src.battle.utils import compute_move_effective_power


@dataclass
class OrderEvaluation:
    battleOrder: BattleOrder
    evaluation: float
    is_random_order: bool = False
    model_i: Optional[int] = None
    pass


# evaluators are tasked with taking as input a battle state and an evaluation of each possible move and modifying it.
# This process can be pipelined
# Let's assume that a pokémon should attack with strong moves, an evaluator could increase the evaluation previously
# assigned to those moves.
class OrderEvaluator(ABC):
    @abstractmethod
    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations: Optional[List[OrderEvaluation]] = None) \
            -> List[OrderEvaluation]:
        pass


# returns a fixed evaluation. Used to start a pipeline of evaluators.
class FixedOrderEvaluator(OrderEvaluator):
    def __init__(self, fixed_value=1) -> None:
        super().__init__()
        self.fixed_evaluation = fixed_value

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        available_orders = battle.available_moves + battle.available_switches
        return [OrderEvaluation(battleOrder=BattleOrder(order=order), evaluation=self.fixed_evaluation) for order in
                available_orders]


def _compute_avg_power(active_pokemon, opponent_active_pokemon, move) -> float:
    p = compute_move_effective_power(active_pokemon, opponent_active_pokemon, move) * move.accuracy
    return p


# do powerful moves
class PowerEvaluator(OrderEvaluator):

    def __init__(self, power_base=70) -> None:
        super().__init__()
        self.power_base = power_base

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        active_pokemon: Gen8Pokemon = battle.active_pokemon
        opponent_active_pokemon: Gen8Pokemon = battle.opponent_active_pokemon
        updated_evaluations = copy.deepcopy(initial_evaluations)
        for e in updated_evaluations:
            order = e.battleOrder.order
            if isinstance(order, Move) and order.category != "status":
                power_factor = _compute_avg_power(active_pokemon, opponent_active_pokemon, order) / self.power_base
                e.evaluation = e.evaluation * power_factor

        return updated_evaluations


# attempt to boost stats if at high hp
class BoostOrderEvaluator(OrderEvaluator):
    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        updated_evaluations = copy.deepcopy(initial_evaluations)
        active_pokemon: Gen8Pokemon = battle.active_pokemon
        tot_boosted = sum([x for x in active_pokemon.boosts.values()])

        for e in updated_evaluations:
            order = e.battleOrder.order
            if isinstance(order, Move):
                if order.boosts is not None:
                    tot_move_boost = sum([x for x in order.boosts.values()])
                    e.evaluation *= (tot_move_boost + 1) / (max(0, tot_boosted) + 1)
                    e.evaluation *= active_pokemon.current_hp_fraction
        return updated_evaluations


# chance swap chance
class BoostSwapEvaluator(OrderEvaluator):
    def __init__(self, swap_boost_factor=0.7) -> None:
        super().__init__()
        self.swap_boost_factor = swap_boost_factor

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        updated_evaluations = copy.deepcopy(initial_evaluations)
        for e in updated_evaluations:
            order = e.battleOrder.order
            if isinstance(order, Pokemon):
                e.evaluation = e.evaluation * self.swap_boost_factor

        return updated_evaluations


# swap to pokemons that have effective moves against the opponent
class BoostSwapToPowerful(OrderEvaluator):

    def __init__(self, base_max_pow=120) -> None:
        super().__init__()
        self.base_max_pow = base_max_pow

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:

        updated_evaluations = copy.deepcopy(initial_evaluations)
        for e in updated_evaluations:
            order = e.battleOrder.order
            if isinstance(order, Pokemon):
                moves = order.moves.values()
                max_out_pow = max(
                    [_compute_avg_power(battle.active_pokemon, battle.opponent_active_pokemon, m) for m in moves]
                )
                e.evaluation = e.evaluation * max_out_pow / self.base_max_pow
        return updated_evaluations


# boosts the effect of another evaluator by a chosen amount. Used to pipeline.
class BoostEvaluator(OrderEvaluator):

    def __init__(self, ev_to_boost: OrderEvaluator, boost: float) -> None:
        super().__init__()
        self.ev_to_boost = ev_to_boost
        self.boost = boost

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        updated_evaluations = self.ev_to_boost.evaluate_orders(battle, initial_evaluations)
        for ie, ue in zip(initial_evaluations, updated_evaluations):
            diff = ie.evaluation - ue.evaluation
            boosted_evaluation = ie.evaluation - diff * self.boost
            ue.evaluation = boosted_evaluation

        return updated_evaluations


class TmpPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        pass


class DeepEvaluator(OrderEvaluator):
    def __init__(self, model: torch.nn.Module, encoder: Encoder) -> None:
        super().__init__()
        self.__player = TmpPlayer()
        self.model = model
        self.encoder = encoder

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        encoding = self.encoder.encode_battle(battle)
        inp = torch.Tensor([encoding]).float()
        ps = self.model(inp)[0]
        order_evs = []
        for i, p in enumerate(ps):
            order, is_random_order = model_pick_to_order(battle, i)
            ev = p.item()
            oe = OrderEvaluation(order, ev, is_random_order=is_random_order, model_i=i)
            order_evs.append(oe)

        if initial_evaluations:
            updated_evaluations = copy.deepcopy(initial_evaluations)
            for u_ev in updated_evaluations:
                for e in order_evs:
                    if u_ev.battleOrder == e.battleOrder:
                        u_ev.evaluation = u_ev.evaluation * e.evaluation
                        return updated_evaluations

        return order_evs


# evaluator using a trained model
class DeepEvaluatorOut(OrderEvaluator):
    def __init__(self) -> None:
        super().__init__()
        deep_encoder = AVAILABLE_ENCODERS[SELECTED_DEEP_ENCODER]
        deep_model = load_latest_model_or_create_new(deep_encoder)
        self.__ev = DeepEvaluator(model=deep_model, encoder=deep_encoder)

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations: Optional[List[OrderEvaluation]] = None) -> \
            List[
                OrderEvaluation]:
        return self.__ev.evaluate_orders(battle, initial_evaluations)


# pipeline for evaluator
class PipelineEvaluator(OrderEvaluator):
    def __init__(self, evaluators: List[OrderEvaluator]) -> None:
        super().__init__()
        self.evaluators = evaluators

    def evaluate_orders(self, battle: AbstractBattle, initial_evaluations=None) -> List[OrderEvaluation]:
        evs = FixedOrderEvaluator().evaluate_orders(battle)
        for ev in self.evaluators:
            evs = ev.evaluate_orders(battle, evs)
        return evs


# declare available evaluators here
AVAILABLE_EVALUATORS = {
    "deep": DeepEvaluatorOut(),
    "s2p": BoostSwapToPowerful(),
    "pwr": PowerEvaluator(),
    "up": BoostOrderEvaluator(),
    "swap": BoostSwapEvaluator()
}


# makes an evaluator pipelining a series of predefined evaluators
# with configurable influence p_i. Each will influence the input_evaluation with the
# output_evaluation = input_evaluation + (i_evaluation - input_evaluation)*p_i
def make_evaluator(selections: Dict[str, float]):
    evaluators = []
    for selected_ev_name, boost in selections.items():
        selected_ev = AVAILABLE_EVALUATORS[selected_ev_name]
        boosted_ev = BoostEvaluator(selected_ev, boost=boost)
        evaluators.append(boosted_ev)

    return PipelineEvaluator(evaluators=evaluators)
