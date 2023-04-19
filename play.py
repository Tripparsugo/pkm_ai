from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration
import asyncio
import time
from poke_env.player import Player, RandomPlayer

from player import DeepPlayer
from src.battle.players import make_standard_player

start = time.time()
my_player_config1 = PlayerConfiguration("bot", None)
my_server_config = ServerConfiguration(
    "0.0.0.0:8000",
    "0.0.0.0:8000/action.php?"
)

# player = DeepPlayer(
#     battle_format="gen8randombattle",
#     player_configuration=my_player_config1,
#     server_configuration=my_server_config
# )

# player = RandomPlayer(
#     battle_format="gen8randombattle",
#     player_configuration=my_player_config1,
#     server_configuration=my_server_config
# )

player = make_standard_player(player_configuration=my_player_config1,
                              server_configuration=my_server_config, use_soft=False)


async def main():
    await player.accept_challenges(None, 3)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
