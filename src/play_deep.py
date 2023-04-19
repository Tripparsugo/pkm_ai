import json

from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration
import asyncio
import time

from config import DEFAULT_SERVER_CONFIG, GEN_LOC
from src.battle.encoders import SimpleEncoder, SimplestEncoder
from src.battle.players import make_standard_player, make_latest_deep_player, make_player

start = time.time()
my_player_config1 = PlayerConfiguration("b1", None)
my_server_config = ServerConfiguration(
    "0.0.0.0:8000",
    "0.0.0.0:8000/action.php?"
)

my_player_config2 = PlayerConfiguration("b2", None)

encoder = SimpleEncoder()
player1 = make_latest_deep_player(player_configuration=my_player_config1,
                                  server_configuration=DEFAULT_SERVER_CONFIG,
                                  use_soft=False,
                                  )

gen_config = None
with open(GEN_LOC) as f:
    tmp = f.read()
    gen_config = json.loads(tmp)

player2 = make_player(gen_config,
                      player_configuration=my_player_config2,
                      server_configuration=DEFAULT_SERVER_CONFIG)


async def main():
    t1 = player1.accept_challenges(None, 3)
    t2 = player2.accept_challenges(None, 3)
    await asyncio.gather(t1, t2)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
