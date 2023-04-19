from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration
import asyncio
import time
from poke_env.player import RandomPlayer

from config import SELECTED_DEEP_ENCODER
from src.battle.encoders import SimplestEncoder, SimpleEncoder
from src.battle.model import get_model_paths
from src.battle.players import make_latest_deep_player, make_standard_player, make_deep_player


async def deep_vs_random(i, model_path, n_battles=100):
    my_player_config1 = PlayerConfiguration(f"A_{i}", None)
    my_player_config2 = PlayerConfiguration(f"B_{i}", None)
    my_server_config = ServerConfiguration(
        "0.0.0.0:8000",
        "0.0.0.0:8000/action.php?"
    )

    encoder = SELECTED_DEEP_ENCODER

    # We create two players.

    deep_player = make_deep_player(model_path, player_configuration=my_player_config1,
                                   server_configuration=my_server_config,
                                   encoder=encoder,
                                   use_soft=False
                                   )

    player2 = make_standard_player(player_configuration=my_player_config2,
                                   server_configuration=my_server_config,
                                   use_soft=False
                                   )

    # random_player = RandomPlayer(
    #     battle_format="gen8randombattle",
    #     player_configuration=my_player_config2,
    #     server_configuration=my_server_config
    # )

    await deep_player.battle_against(player2, n_battles=n_battles)
    return deep_player.n_won_battles


if __name__ == '__main__':
    model_paths = get_model_paths()
    num_models = len(model_paths)
    n_battles = 30
    res = []
    for i in range(0, num_models, 3):
        print(f"{i}/{num_models}")
        model_path = model_paths[i]
        wins = asyncio.get_event_loop().run_until_complete(deep_vs_random(i, model_path, n_battles=n_battles))
        win_frac = wins / n_battles
        res.append((wins, n_battles, win_frac, i))

    for wins, n_battles, win_frac, i in res:
        print(f"{wins}/{n_battles} won. WR = {win_frac}. Model {i}")
