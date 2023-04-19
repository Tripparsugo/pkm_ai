from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration, LocalhostServerConfiguration
import asyncio
import time
from poke_env.player import Player, RandomPlayer

import vectorization
from src.battle.encoders import SimpleEncoder
from src.battle.players import make_latest_deep_player


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        e = SimpleEncoder().encode_battle(battle)
        e2 = SimpleEncoder().encode_battle(battle, valid=False)
        if battle.available_moves:
            # Finds the best move among available ones

            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            battle_order = self.create_order(best_move)
            v = vectorization.vectorizeTurnInfo(battle, battle_order, True)
            v2 = vectorization.vectorizeTurnInfo(battle, battle_order, False)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()
    my_player_config1 = PlayerConfiguration("my_username1", None)
    my_player_config2 = PlayerConfiguration("my_username2", None)
    my_server_config = ServerConfiguration(
        "0.0.0.0:8000",
        "0.0.0.0:8000/action.php?"
    )

    # We create two players.

    deep_player = make_latest_deep_player(player_configuration=my_player_config1,
                                          server_configuration=my_server_config)

    # random_player = MaxDamagePlayer(
    #     battle_format="gen8randombattle",
    #     player_configuration=my_player_config1,
    #     server_configuration=my_server_config
    # )
    max_damage_player = RandomPlayer(
        battle_format="gen8randombattle",
        player_configuration=my_player_config2,
        server_configuration=my_server_config
    )

    # Now, let's evaluate our player
    await max_damage_player.battle_against(deep_player, n_battles=10)

    # await max_damage_player.stop_listening()
    # await deep_player.stop_listening()
    # await deep_player.listen()
    # await max_damage_player.listen()

    await max_damage_player.battle_against(deep_player, n_battles=10)

    print(
        "Max damage player won %d / 10 battles [this took %f seconds]"
        % (
            max_damage_player.n_won_battles, time.time() - start
        )
    )


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
