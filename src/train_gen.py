import asyncio
import copy
import pathlib
import random
from dataclasses import dataclass
from typing import List, Callable
from elosports.elo import Elo
from alive_progress import alive_bar
import json

from config import DEFAULT_SERVER_CONFIG, GEN_SELECTED_EVALUATORS, AVG_P_MATCHES, POPULATION_SIZE, LD_CYCLES, \
    LD_INSTANCES, MODEL_DIR, DATA_DIR, TMP_DIR, GEN_LOC, KILL_PICK_CHANCE, GEN_PICK_CHANCE
from src.battle.encoders import SimpleEncoder
from src.battle.evaluators import PipelineEvaluator, PowerEvaluator, BoostEvaluator, BoostSwapToPowerful, \
    BoostOrderEvaluator, BoostSwapEvaluator, DeepEvaluator, make_evaluator
from src.battle.model import load_latest_model_or_create_new
from src.battle.pickers import BestPicker
from src.battle.players import ConcretePlayer
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration
import numpy as np
import pandas as pd

genes = List[float]


@dataclass
class PlayerRecord:
    player: ConcretePlayer
    genes: genes
    creation_era: int = None
    elo: int = 1000
    death_era = None


def make_gen_player(genes, server_configuration, player_configuration):
    p = {x: genes[i] for i, x in enumerate(GEN_SELECTED_EVALUATORS)}
    evaluator = make_evaluator(p)
    picker = BestPicker()
    return ConcretePlayer(evaluator=evaluator, picker=picker,
                          server_configuration=server_configuration, player_configuration=player_configuration)


async def play_round(prs: List[PlayerRecord], league: Elo, matches: int = 100, on_match_end=None):
    for _ in range(0, matches):
        potential_picks = list(range(0, len(prs)))
        pick1_i = random.randint(0, len(potential_picks) - 1)
        pick1 = potential_picks[pick1_i]
        potential_picks.remove(pick1)
        pick2_i = random.randint(0, len(potential_picks) - 1)
        pick2 = potential_picks[pick2_i]
        pr1 = prs[pick1]
        pr2 = prs[pick2]
        wins_pre = pr1.player.n_won_battles
        await pr1.player.battle_against(pr2.player)
        if on_match_end:
            on_match_end()
        p1_won = pr1.player.n_won_battles > wins_pre
        winner, loser = (pr1.player.username, pr2.player.username) if p1_won else (
            pr2.player.username, pr1.player.username)
        league.gameOver(winner=winner, loser=loser, winnerHome=False)
        pr1.elo = league.ratingDict[pr1.player.username]
        pr2.elo = league.ratingDict[pr2.player.username]


# takes as input the genes of n parents and combines into the genes of
def combine_genes(gens: List[genes], mutation_chance: float = 0.3, mutation_size: float = 0.2) -> genes:
    new_gen = [x / len(gens) for x in gens[0]]
    # avg of all parent genes
    for gen in gens[1:]:
        for i, v in enumerate(gen):
            new_gen[i] = new_gen[i] + (gen[i] / len(gens))
    # alter genes
    for i, v in enumerate(new_gen):
        mutation: float = random.random() * (mutation_size - mutation_size / 2) if random.random() < mutation_chance \
            else 0

        new_v = max(0., v + mutation)
        new_gen[i] = new_v
    return new_gen


# suggest a player to kill  based on their performance in the league
def pick_moribund(prs: List[PlayerRecord], a=0.5, b=60) -> PlayerRecord:
    by_elo = sorted(prs, key=lambda pr: pr.elo)
    # newborns should have a chance to play before they are killed
    by_elo = [x for x in by_elo if x.player.n_finished_battles > 0]

    for i in range(0, len(by_elo) - 1):
        pr = by_elo[i]
        # pr2 = by_elo[i + 1]
        if random.random() < a:
            return pr

    return by_elo[-1]


# selects a new player to cross based on the league performance
def pick_progenitor(prs: List[PlayerRecord], a=0.5, b=60) -> PlayerRecord:
    by_elo = sorted(prs, key=lambda pr: pr.elo, reverse=True)
    for i in range(0, len(by_elo) - 1):
        pr = by_elo[i]
        # pr2 = by_elo[i + 1]
        if random.random() < a:
            return pr

    return by_elo[-1]


def pick_progenitors(prs: List[PlayerRecord], a=0.5, b=60, c=0.5):
    prs = copy.copy(prs)
    new_p = pick_progenitor(prs, a=a, b=b)
    prs.remove(new_p)
    progenitors = [new_p]

    while random.random() < c and len(prs) > 0:
        new_p = pick_progenitor(prs, a=a, b=b)
        prs.remove(new_p)
        progenitors.append(new_p)

    return progenitors


# returns deaths and lifes in life_death cycle
def ld_instance(prs: List[PlayerRecord], get_player_config: Callable[[], PlayerConfiguration], kill_pick_chance=0.5,
                gen_pick_chance=60, c=0.5) -> (
        List[PlayerRecord], List[PlayerRecord]):
    dead_player = pick_moribund(prs, a=kill_pick_chance, b=60)
    progenitors = pick_progenitors(prs, a=gen_pick_chance, b=60)
    new_genes = combine_genes([progenitor.genes for progenitor in progenitors])
    my_player_config = get_player_config()
    new_player = make_gen_player(new_genes, server_configuration=DEFAULT_SERVER_CONFIG,
                                 player_configuration=my_player_config)
    new_record = PlayerRecord(player=new_player, genes=new_genes)
    return dead_player, new_record


if __name__ == '__main__':
    count = 0
    era = 0
    matches = AVG_P_MATCHES * POPULATION_SIZE
    eloLeague = Elo(k=20, homefield=0)
    current_players: List[PlayerRecord] = []
    all_players: List[PlayerRecord] = []


    def make_player_config() -> PlayerConfiguration:
        global count
        my_player_config = PlayerConfiguration(f"user_{count}", None)
        count = count + 1
        return my_player_config


    with alive_bar(POPULATION_SIZE, dual_line=True, title='Building population', force_tty=True) as bar:
        for i in range(0, POPULATION_SIZE):
            my_player_config = make_player_config()
            params_n = len(GEN_SELECTED_EVALUATORS)
            genes = [random.random() for _ in range(0, params_n)]
            player = make_gen_player(genes, server_configuration=DEFAULT_SERVER_CONFIG,
                                     player_configuration=my_player_config)
            pr = PlayerRecord(player=player, creation_era=era, genes=genes)
            current_players.append(pr)
            user = pr.player.username
            eloLeague.addPlayer(user)
            bar()

    all_players = copy.copy(current_players)
    n_matches = AVG_P_MATCHES * POPULATION_SIZE * LD_CYCLES

    with alive_bar(n_matches, dual_line=True, title='Tournament', force_tty=True) as bar:
        def on_match_end():
            bar()


        for i in range(0, LD_CYCLES):
            asyncio.get_event_loop().run_until_complete(
                play_round(current_players, matches=matches, league=eloLeague, on_match_end=on_match_end))
            for j in range(0, LD_INSTANCES):
                dead_player, new_player = ld_instance(current_players, make_player_config,
                                                      kill_pick_chance=KILL_PICK_CHANCE,
                                                      gen_pick_chance=GEN_PICK_CHANCE)
                dead_player.death_era = era
                new_player.creation_era = era
                eloLeague.addPlayer(new_player.player.username)
                current_players.remove(dead_player)
                current_players.append(new_player)
                all_players.append(new_player)
            era = era + 1

    data = [p.genes + [p.creation_era] + [p.death_era] + [p.elo] + [p.player.n_finished_battles] for p in all_players]
    columns = GEN_SELECTED_EVALUATORS + ["creation_era", "death_era", "elo", "matches_n"]
    df = pd.DataFrame(data=data, columns=columns)
    print(df.head(n=10))
    df.to_csv(f"{DATA_DIR}/gen.csv")
    alive_df = df[df["death_era"].isnull()]
    best_ps = alive_df.loc[alive_df["elo"].idxmax()]
    best_config = {se: best_ps[se] for se in GEN_SELECTED_EVALUATORS}
    c_path = pathlib.Path(GEN_LOC)
    c_path.parent.mkdir(parents=False, exist_ok=True)
    j_c = json.dumps(best_config)
    with open(c_path, 'w') as f:
        f.write(j_c)
