import pandas as pd
from rec2vec.util.encoding_detector import get_encoding
from rec2vec.util.load_config import load_config
from rec2vec.util.Graph import Graph
import random
from gensim.models import Word2Vec
from scripts.test import predict_and_test

config = load_config()
config_path = './rec2vec/configs/graph_config.yaml'
data_path = './data/test_user_ratings.csv'

df = pd.read_csv(filepath_or_buffer=data_path,
                 sep=config['data']['separator'],
                 encoding=get_encoding(file=data_path))


number_paths = [2, 3]
length_path = [3, 4, 5]
alpha = [0.1, 0.2, 0.3]
window_size = [5]

g = Graph(config_path)
rand = random.Random(0)

node_dict = g.get_node_dict()

acc = 0
mse = float('inf')

best_config = []
with open('./output/hyperparameter_tuning_report.csv', 'w') as f:
    for paths in number_paths:
        for lengths in length_path:
            for alphas in alpha:
                for windows in window_size:
                    corpus = g.build_deepwalk_corpus(num_paths=paths, path_length=lengths,
                                                     alpha=alphas, rand=rand)
                    model = Word2Vec(sentences=corpus, window=windows, min_count=0, workers=8)
                    res_acc, res_cm, res_mse = predict_and_test(data_path, 'users:userID;movies:movieID', 'ratings:rating', config, model, node_dict)
                    if mse > float(res_mse):
                        best_config = [paths, lengths, alphas, windows, res_acc, res_cm, res_mse]
                    mse = float(res_mse)
                    f.write(f'{paths};{lengths};{alphas};{windows};{res_acc};{res_cm};{res_mse}\n')
                    print(f'{[paths, lengths, alphas, windows, res_acc, res_cm, res_mse]}')

