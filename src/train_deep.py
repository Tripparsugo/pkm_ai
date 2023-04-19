import os
import random
from typing import List, Dict

import torch
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration
import asyncio
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from config import TMP_DIR
from config import SELECTED_DEEP_ENCODER, N_BATTLES, LOOPS
from src.battle.encoders import AVAILABLE_ENCODERS
from src.battle.model import load_latest_model_or_create_new, save_model
from src.battle.players import make_latest_deep_player
from src.battle.record import BattleRecorder, ConcreteRewardAssigner, TurnRecord
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
from lightning.pytorch.callbacks import TQDMProgressBar
from poke_env.environment import AbstractBattle

train_frac = 0.9

train = True
encoder = AVAILABLE_ENCODERS[SELECTED_DEEP_ENCODER]
loop_c = 0


def save_turns(turns: Dict[AbstractBattle, List[TurnRecord]], log: bool):
    if not log:
        return

    data = []
    battles = list(turns.keys())

    tmp: List[TurnRecord] = [y for x in zip(turns[battles[0]], turns[battles[1]]) for y in x]

    for t in tmp:
        if t is None:
            continue
        data.append(
            {
                "StateValue": t.state_value,
                "Reward": t.reward,
                "Order": "" if t.order is None else str(t.order.order),
                "Username": t.player_name,
                "TurnN": t.turn_n
            }
        )

    df = pd.DataFrame(data=data)
    df.to_csv(f"{TMP_DIR}/turns.csv", index=False)


async def main():
    my_player_config1 = PlayerConfiguration(f"usernA{loop_c}", None)
    my_player_config2 = PlayerConfiguration(f"userB{loop_c}", None)
    my_server_config = ServerConfiguration(
        "0.0.0.0:8000",
        "0.0.0.0:8000/action.php?"
    )

    player1 = make_latest_deep_player(player_configuration=my_player_config1,
                                      server_configuration=my_server_config,
                                      use_soft=True,
                                      )

    player2 = make_latest_deep_player(player_configuration=my_player_config2,
                                      server_configuration=my_server_config,
                                      use_soft=True,
                                      )

    br = BattleRecorder(encoder=encoder, ra=ConcreteRewardAssigner(decay=0.6,
                                                                   window_size=5,
                                                                   faint_weight=0.1,
                                                                   type_weight=0.2,
                                                                   status_weight=0.1),
                        )
    val_frac = 1 - train_frac
    n_train = int(N_BATTLES * train_frac)
    n_val = int(N_BATTLES * val_frac)

    battle_count = 0

    def register_turn(player, battle, order, order_i):
        # we have to skip this because this state is bugged and the battle is not updated until the following turn
        if battle.active_pokemon is not None and battle.active_pokemon.current_hp_fraction == 0:
            return
        br.push_turn(player, battle, order, order_i)

    def register_battle_end(player, battle, order, order_i):
        br.push_turn(player, battle, order, order_i)
        nonlocal battle_count
        battle_count = battle_count + 1
        print(f"battle {battle_count}/{N_BATTLES * 2}")

    player1.turn_callback = register_turn
    player1.battle_end_callback = register_battle_end
    player2.turn_callback = register_turn
    player2.battle_end_callback = register_battle_end

    await player1.battle_against(player2, n_battles=n_train)
    d = br.battle_to_turns.values()
    train_turns = [t for ts in d for t in ts]
    train_l = len(train_turns)
    save_turns(br.battle_to_turns, True)

    br.reset()

    await player1.battle_against(player2, n_battles=n_val)
    d = br.battle_to_turns.values()
    validation_turns = [t for ts in d for t in ts]
    turns = train_turns + validation_turns

    if not train:
        return

    # load model
    model = load_latest_model_or_create_new(encoder=encoder)
    print("preparing data")
    data = []
    for t in turns:
        if t.reward is None:
            continue
        x = torch.Tensor(t.encoding).float()
        y = torch.tensor([t.order_i, t.reward] + [0] * (22 - 2))
        data.append((x, y))

    train_data = data[0:train_l]
    validation_data = data[train_l:]

    # using workers slows up the process by x30, not sure why
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32)

    # train
    logger = TensorBoardLogger(save_dir=f"{TMP_DIR}", name="train_deep")
    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5),
                                    TQDMProgressBar(refresh_rate=1)
                                    ],
                         logger=logger,
                         log_every_n_steps=10,
                         )

    # torch.autograd.set_detect_anomaly(True)
    print("training")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    # save trained model
    save_model(model)
    print("saved model")


if __name__ == '__main__':
    for i in range(0, LOOPS):
        asyncio.get_event_loop().run_until_complete(main())
        print(f"loop{i + 1}/{LOOPS}")
        loop_c = loop_c + 1
