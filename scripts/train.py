import gensim.models

from rec2vec.util.Graph import Graph
from gensim.models import Word2Vec
from sys import exit
from pickle import dump
from rec2vec import logger

import random
import argparse


def save_model(path: str, model: gensim.models.Word2Vec):
    logger.trace(f'save_model({path}, {model})')
    logger.info('saving model...')
    filehandler = open(file=path, mode='wb')
    dump(obj=model, file=filehandler)
    filehandler.close()


def train(args: argparse.Namespace, g: Graph = None, save: bool = True):
    """
    Trains a Word2Vec model with the arguments the user provided.

    :param args:    see help for detailed information on the user input
    :param g:       graph to be used for training
    :param save:    boolean indicating whether to save the trained model or not
    :return:
    """
    logger.trace(f'train({args})')
    logger.info('Training with the following arguments:\n'
                f'number of paths:\t{args.number_paths}\n'
                f'path length:\t\t{args.length_path}\n'
                f'alpha:\t\t\t{args.alpha}\n'
                f'seed:\t\t\t{args.seed}\n'
                f'window size:\t\t{args.window_size}\n'
                f'workers:\t\t{args.workers}\n'
                f'save path:\t\t{args.save_path}\n'
                f'config path:\t\t{args.config_path}\n')

    if g is None:
        logger.info('constructing graph...')
        g = Graph(config_path=args.config_path)
        logger.info('graph constructed successfully')

    rand = random.Random(args.seed)

    logger.info('constructing corpus...')
    corpus = g.build_deepwalk_corpus(num_paths=args.number_paths, path_length=args.length_path,
                                     alpha=args.alpha, rand=rand)
    logger.info('corpus constructed successfully')

    logger.info('creating model...')
    model = Word2Vec(sentences=corpus, window=args.window_size, min_count=0, workers=args.workers)
    logger.info('model created successfully')

    if save:
        save_model(path=args.save_path, model=model)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train the Word2Vec model')
    parser.add_argument('-np', '--number-paths', default=10, type=int, help='Number of paths for each node')
    parser.add_argument('-lp', '--length-path', default=40, type=int, help='Number of steps per path')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('-ws', '--window-size', default=5, type=int, help='Window size for skipgram')
    parser.add_argument('-wo', '--workers', default=8, type=int, help='Number of workers')
    parser.add_argument('-sp', '--save-path', default='./models/word2vec.obj', type=str, help='Path where trained model shall be stored')
    parser.add_argument('-cp', '--config-path', default='./rec2vec/configs/graph_config.yaml', type=str, help='Path to custom config')
    parser.add_argument('-a', '--alpha', default=0, type=float, help='Chance for path to be reset to start')
    args = parser.parse_args()
    train(args=args)


if __name__ == '__main__':
    exit(main())
#%%
