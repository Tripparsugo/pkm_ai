from gym.utils.env_checker import check_env
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration

from src.battle.environments import RLPEnvPlayer
from poke_env.player import RandomPlayer

if __name__ == '__main__':
    my_player_config1 = PlayerConfiguration("my_username1", None)
    my_player_config2 = PlayerConfiguration("my_username2", None)
    my_server_config = ServerConfiguration(
        "0.0.0.0:7777",
        "0.0.0.0:7777/action.php?"
    )

    opponent = RandomPlayer(battle_format="gen8randombattle",
                            server_configuration=my_server_config, player_configuration=my_player_config2)
    test_env = RLPEnvPlayer(opponent=opponent, battle_format="gen8randombattle", start_challenging=True,
                            server_configuration=my_server_config, player_configuration=my_player_config1)
    check_env(test_env)
    test_env.close()
