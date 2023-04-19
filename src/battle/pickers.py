import random
from abc import ABC, abstractmethod
from typing import List

from src.battle.evaluators import OrderEvaluation
from poke_env.player.battle_order import BattleOrder
from poke_env.environment.move import Move

from src.battle.utils import softmax


# a picker is tasked with picking an action out of the ones proposed using their evaluations.
# usually the move with the highest value should be used. In other cases softmax might be used to explore the
# space
class Picker(ABC):
    @abstractmethod
    def pick_order(self, evs: List[OrderEvaluation]) -> int:
        pass


class BestPicker(Picker):
    def pick_order(self, evs: List[OrderEvaluation]):
        best = evs[0]
        best_i = 0
        for i, ev in zip(range(1, len(evs)), evs[1:]):
            # we need to start from
            if ev.evaluation > best.evaluation:
                best = ev
                best_i = i

        return best_i


class RandomPicker(Picker):
    def pick_order(self, evs: List[OrderEvaluation]):
        r = random.randint(0, len(evs))
        return r


class RandomAttackPicker(Picker):
    def pick_order(self, evs: List[OrderEvaluation]):
        move_orders = [x for x in evs if isinstance(x.battleOrder.order, Move)]
        rp = RandomPicker()
        if len(move_orders) > 0:
            return rp.pick_order(move_orders)

        return rp.pick_order(evs)


class SoftmaxPicker(Picker):
    def pick_order(self, evs: List[OrderEvaluation]):
        vs = [x.evaluation for x in evs]
        ss = softmax(vs)
        r = random.random()
        tmp = 0
        i = 0
        while tmp <= r and i < len(evs):
            tmp = tmp + ss[i]
            i = i + 1
        return i - 1


# declare available pickers here
AVAILABLE_PICKERS = {
    "softmax": SoftmaxPicker(),
    "best": BestPicker(),
    "random": RandomPicker()
}
