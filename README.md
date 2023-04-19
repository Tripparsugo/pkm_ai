## DESCRIPTION

This is a collection of scripts to train, test and analyze the matches of an AI player for Pokémon.

## REQUIREMENTS

The scripts are in python and need Python >= 3.8.9 in order to be run:

To install the required libraries:

```
pip install -r requirements.txt
```

The scripts need to connect to a running version of the pokémon-showdown simulator. This requires an installation of
node on the system. To run the simulator:

```
git clone https://github.com/hsahovic/Pokemon-Showdown.git
cd Pokemon-Showdown
node pokemon-showdown start --no-security
```

This will start the simulator on http://localhost:8000, if using another simulator change the configuration in:

```
config.py
```

## TRAINING A PLAYER

to train the AI player run:

```
python ./src/train_deep.py
```

this will save a model for each training iteration.

After the initial training a refining step using a genetic algorithm can be performed by running:

```
train_gen.py
```

configuration of the training parameters in ```config.py```.

## TESTING THE AI


## PLAYING AGAINST THE AI




