from gym.utils.env_checker import check_env
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration

from src.battle.model import create_model, get_dqn, get_env
from src.battle.players import make_standard_player
from src.battle.environments import RLPEnvPlayer
from poke_env.player import RandomPlayer

if __name__ == '__main__':
    my_player_config1 = PlayerConfiguration("user1", None)
    my_player_config2 = PlayerConfiguration("user2", None)
    my_player_config3 = PlayerConfiguration("user3", None)
    my_player_config4 = PlayerConfiguration("user4", None)
    my_server_config = ServerConfiguration(
        "0.0.0.0:8000",
        "0.0.0.0:8000/action.php?"
    )

    opponent = make_standard_player(server_configuration=my_server_config, player_configuration=my_player_config2,
                                    use_soft=True)

    standard_opponent = make_standard_player(server_configuration=my_server_config,
                                             player_configuration=my_player_config3,
                                             use_soft=False)

    random_opponent = RandomPlayer(server_configuration=my_server_config, player_configuration=my_player_config4)

    test_env = RLPEnvPlayer(opponent=opponent, battle_format="gen8randombattle", start_challenging=True,
                            server_configuration=my_server_config, player_configuration=my_player_config1)

    train_env = get_env(opponent)

    n_action = train_env.action_space.n
    model = create_model()
    dqn = get_dqn(model, n_action)
    model.save("models/1")

    eval_env = get_env(random_opponent)
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=50, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=True, opponent=standard_opponent)
    print("Results against standard player:")
    dqn.test(eval_env, nb_episodes=50, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
