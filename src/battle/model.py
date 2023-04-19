import pathlib
from typing import List, Union, Awaitable, Optional, Tuple
from pathlib import Path
from poke_env.environment import AbstractBattle
from config import MODEL_DIR
from config import SELECTED_DEEP_ENCODER
from src.battle.encoders import SimpleEncoder, Encoder
from src.battle.environments import RLPEnvPlayer
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.python.keras.optimizer_v2 import adam
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player import Player, ForfeitBattleOrder, BattleOrder
import torch
import lightning.pytorch as pl
import torch.nn.functional as F


class PokeBrain(pl.LightningModule):
    def __init__(self, d_in, d_out):
        super(PokeBrain, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, 200)
        self.f1 = torch.nn.ELU()
        self.linear2 = torch.nn.Linear(200, 100)
        self.f2 = torch.nn.ELU()
        self.linear3 = torch.nn.Linear(100, d_out)
        self.f3 = torch.nn.Softsign()

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=False)
        x = self.linear2(x)
        x = F.relu(x, inplace=False)
        x = self.linear3(x)
        x = F.softsign(x)
        return x

    def training_step(self, train_batch, batch_idx):
        # train_batch = (torch.rand(64, 324), torch.rand(64, 22))
        loss_f = torch.nn.L1Loss()
        xs, ds = train_batch
        logits = self.forward(xs)
        ys = []
        for l, d in zip(logits.detach(), ds):
            y = l.clone().numpy()
            i = int(d[0].item())
            reward = d[1].item()
            y[i] = reward
            ys.append(y)

        ys = torch.tensor(ys)

        loss = loss_f(logits, ys)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss_f = torch.nn.L1Loss()
        xs, ds = val_batch
        logits = self.forward(xs)
        ys = []
        for l, d in zip(logits.detach(), ds):
            y = l.clone().numpy()
            i = int(d[0].item())
            reward = d[1].item()
            y[i] = reward
            ys.append(y)

        ys = torch.tensor(ys)

        loss = loss_f(logits, ys)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {'optimizer': optimizer,
                'scheduler': scheduler,
                'monitor': "val_loss",
                }


def create_model(encoder: Encoder):
    model = PokeBrain(d_in=encoder.encoding_size(), d_out=22)
    return model


def get_env(opponent, start_challenging=False):
    env = RLPEnvPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=start_challenging
    )
    return env


def load_model(path):
    model = torch.load(path)
    return model


def load_latest_model_or_create_new(encoder: Encoder = SimpleEncoder()):
    pathlib.Path(f"{MODEL_DIR}/{SELECTED_DEEP_ENCODER}").mkdir(parents=True, exist_ok=True)
    model_paths = get_model_paths()
    if len(model_paths) == 0:
        model = create_model(encoder=encoder)
        save_model(model)
        return model

    latest_path = f"{MODEL_DIR}/{SELECTED_DEEP_ENCODER}/{len(model_paths) - 1}"
    model = load_model(latest_path)
    return model


def get_model_paths():
    p = Path(f"{MODEL_DIR}/{SELECTED_DEEP_ENCODER}")
    #  not recursive.
    paths = [f for f in p.iterdir()]
    return paths


def save_model(model):
    n = len(get_model_paths())
    path = f"{MODEL_DIR}/{SELECTED_DEEP_ENCODER}/{n}"
    torch.save(obj=model, f=path)


def get_dqn(model, n_action):
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    # tf.compat.v1.disable_eager_execution()
    dqn.compile(adam.Adam(learning_rate=0.00025), metrics=["mae"])
    return dqn


def states_to_rewards(battle_states: List[AbstractBattle], won: bool) -> List[float]:
    pass


def state_to_value(battle_state: AbstractBattle) -> float:
    pass


# used to access utility methods of player
class DummyPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> Union[BattleOrder, Awaitable[BattleOrder]]:
        pass


def model_pick_to_order(battle: AbstractBattle, action: int) -> Tuple[BattleOrder, bool]:
    """Converts actions to move orders.

    The conversion is done as follows:

    action = -1:
        The battle will be forfeited.
    0 <= action < 4:
        The actionth available move in battle.available_moves is executed.
    4 <= action < 8:
        The action - 4th available move in battle.available_moves is executed, with
        z-move.
    8 <= action < 12:
        The action - 8th available move in battle.available_moves is executed, with
        mega-evolution.
    12 <= action < 16:
        The action - 12th available move in battle.available_moves is executed,
        while dynamaxing.
    16 <= action < 22
        The action - 16th available switch in battle.available_switches is executed.

    If the proposed action is illegal, a random legal move is performed.

    :param action: The action to convert.
    :type action: int
    :param battle: The battle in which to act.
    :type battle: Battle
    :return: the order to send to the server.
    :rtype: str
    """

    player = DummyPlayer(start_listening=False)
    if action == -1:
        return ForfeitBattleOrder(), False
    elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
    ):
        return player.create_order(battle.available_moves[action]), False
    elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
    ):
        return player.create_order(
            battle.active_pokemon.available_z_moves[action - 4], z_move=True
        ), False
    elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
    ):
        return player.create_order(
            battle.available_moves[action - 8], mega=True
        ), False
    elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
    ):
        return player.create_order(
            battle.available_moves[action - 12], dynamax=True
        ), False
    elif 0 <= action - 16 < len(battle.available_switches):
        return player.create_order(battle.available_switches[action - 16]), False
    else:
        return player.choose_random_move(battle), True


def order_to_model_pick(battle: AbstractBattle, order: BattleOrder) -> List[int]:
    model_picks = []
    for i in range(0, 22):
        potential_order = model_pick_to_order(battle, i)
        if potential_order == order:
            model_picks.append(i)

    if len(model_picks) == 0:
        raise Exception("match not found")

    # r = random.randint(0, len(model_picks))
    # model_pick = model_picks[r]
    return model_picks
