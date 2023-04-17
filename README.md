# Rec2Vec using DeepWalk

This project implements DeepWalk to predict ratings in a graph-based recommender system. It reads the 
graph properties from a configuration file so that it can handle various datasets.

For details, please refer to [this paper](https://arxiv.org/abs/1403.6652). The dataset is taken from
[here](https://github.com/triandicAnt/GraphEmbeddingRecommendationSystem/tree/e05b69c2209c09a6b99c21ecc47e14804e6a5c60/data)
and has been slightly modified (changed separator to `;` and used added ``.csv`` file extension).


There are important aspects that were not implemented for example, parallelization or maximum memory size. Also, 
the system has not yet been tested with other datasets.

----

## How to Run the Project

ll commands listed below are expected to be run from the root directory of this project. 
[Conda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html) is assumed 
to be used as a package manager.

### Install Dependencies

```shell
conda create -n rec2vec python=3.10
conda activate rec2vec
pip install -e .
```

The data can already be found in the `./data/` folder.

### Train Model

```shell
python scripts/train.py
```

The model can be trained (and subsequently saved) by running the command above. For help, execute the command
with the flag ``-h``.
By default, the model will be saved to ``./models/rec2vec.obj``.

### Test Model Performance

```shell
python scripts/test.py
```

The model can be tested by running the command above. For help, execute the command with the flag ``-h``.
By default, the script will produce a report containing the parameters and results (accuracy, mean squared 
error and confusion matrix) in ``./output/``.

### Hyperparameter Tuning

```shell
python scripts/tuning.py
```

There is a short script that allows to tune the model's hyperparameters. It will produce a report in ``./output/``.

### Change Parameters

The script ``./scripts/train.py`` allow a user to train the model with different
arguments. Relevant for the performance are:
- `--number-paths`: How many paths to construct for each node
- `--length-path`: How long each path has to be
- `--seed`: Seed for reproducibility
- `--alpha`: Probability for randomly resetting the path

### Benchmark

Todo: add best results for hyperparameter tuning.

----

## Change Datasource

This application is meant to be used for data that has to with user-item ratings. It uses 
the ``./rec2vec/configs/graph_config.yaml`` to construct the graph. If one wants to try
the application with a different dataset, and there is a change in attributes or data source,
the ``graph_config.yaml`` file has to be changed accordingly. One can also create a different
``.yaml`` file and simply tell the application where to find th config using the
flag ``--config-path`` or ``-cp``.

The application makes certain assumptions. Thus, the config must contain a similar structure (which 
should be doable in a setting that deals with data for recommendations).

The data section is about data input and data output. All input files have to be in one folder
(by default ``./data/``). 
```yaml
data:
  folder: ./data/
  separator: ;
  output_folder: ./output/
  objects:
    node_dict: node_dict.obj
    final_graph: graph.obj
...
```

Vertices of the graph must follow this pattern. Each node (`movies`, `directors`, `actors`, ...) forms a so-called
`node_type`. That simply is a dictionary key to identify nodes of different types if they have the same `id`.

Every node has a ``source``. That is the name of the `.csv` file that contains all entries of that node type.
``column`` indicates the name of the column in which the node's `id` can be found.

Furthermore, an ``id_prefix`` can be added to indicate the `node_type`.

For the node type whose ratings we want to predict, we have to add the ``extended`` section. This means, that
a node is extended by another type of node (``ratings``). We also add a ``range``. `0-5` means that
a ``movie`` can be rated from 0 to 5. There is no support for decimal places. It is important to note that it is not
required to make ``ratings`` a separate node type. It's sufficient to simply extend the so-called `target node type`.

```yaml
...
nodes:
  movies:
    source: movies.csv
    column: id
    id_prefix: m
    extended:
      by: ratings
      range: 0-5
  directors:
    source: movie_directors.csv
    column: directorID
  actors:
    source: movie_actors.csv
    column: actorID
...
```

In the ``edges`` section, we have to define the connections between nodes. Each edge (`movies_actors`, 
`users_ratings`, ...) combines two vertices. It needs a `source` (file that contains the relation) and two
vertices (`vertex1` and `vertex2`, but the order does not matter). Each vertex has a `column` and a `type`
(referring to the `node type`).

Since the ``extending node`` does not have IDs in the original data, we have to use `extending` and `extend_with`
to specify the `base node type` (`movies`) and which column extends this type (`rating`).

```yaml
...
edges:
  movies_actors:
    source: movie_actors.csv
    vertex1:
      column: movieID
      type: movies
    vertex2:
      column: actorID
      type: actors
  users_ratings:
    source: train_user_ratings.csv
    vertex1:
      column: userID
      type: users
    vertex2:
      column: movieID
      type: ratings
      extend_with: rating
      extending: movies
...
```

----



