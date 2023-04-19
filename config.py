from poke_env import ServerConfiguration

from anchor import PROJECT_ROOT

DEFAULT_SERVER_CONFIG = ServerConfiguration(
    "0.0.0.0:8000",
    "0.0.0.0:8000/action.php?"
)
MODEL_DIR = f"{PROJECT_ROOT}/models/deep"
GEN_LOC = f"{PROJECT_ROOT}/models/gen/gen.json"
TMP_DIR = f"{PROJECT_ROOT}/tmp"
DATA_DIR = f"{PROJECT_ROOT}/data"

# ENCODER FOR GAME STATE
# simple|simplest
SELECTED_DEEP_ENCODER = "simplest"

# GEN PARAMS
# evaluators to be used for genetic training
#  ["s2p", "pwr", "deep", "up", "swap"]
GEN_SELECTED_EVALUATORS = ["s2p", "pwr", "up", "swap"]
POPULATION_SIZE = 5
LD_CYCLES = 3
LD_INSTANCES = 2
AVG_P_MATCHES = 5
KILL_PICK_CHANCE = 0.5
GEN_PICK_CHANCE = 0.5

# TRAIN PARAMS
N_BATTLES = 30
LOOPS = 1
RETRAIN = False

# REWARD PARAMS
WINDOW_SIZE = 1
DECAY = 0.5
HP_WEIGHT = 1
FAINT_WEIGHT = 0.3
STATUS_WEIGHT = 0.2
TYPE_WEIGHT = 0.1
