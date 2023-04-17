from pickle import load
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from rec2vec import logger
from rec2vec.util.load_config import load_config
from rec2vec.util.encoding_detector import get_encoding
from rec2vec.predict.prediction_util import predict_from_data
from gensim.models import Word2Vec

import pandas as pd
import argparse
import time


def _report_prediction(args: argparse.Namespace) -> None:
    """
    Predicts ratings for test data, given user input arguments, and produces a report.

    :param args:    user input arguments (path to training data, path to trained model and path to node dict)
    :return:
    """

    logger.trace(f'predict({args})')

    argument_notice = 'Predicting with the following arguments:\n' + \
        f'path to test data:\t{args.data_path}\n' + \
        f'path to model:\t\t{args.model_path}\n' + \
        f'path to node dict:\t{args.node_dict_path}\n' + \
        f'path to report:\t\t{args.report_path}\n' + \
        f'config path:\t\t{args.config_path}\n' + \
        f'target variable:\t{args.target_variable}\n' + \
        f'predictor variable:\t{args.predictor_variable}\n'

    logger.info(argument_notice)

    # Get config that stores information about graph and data sources
    config = load_config(path=args.config_path)

    # Load trained model
    filehandler = open(file=args.model_path, mode='rb')
    model = load(file=filehandler)
    filehandler.close()

    # Load node dictionary
    filehandler = open(file=args.node_dict_path, mode='rb')
    node_dict = load(file=filehandler)
    filehandler.close()

    # Perform training
    acc, cm, mse = predict_and_test(data_path=args.data_path, predictor_variable=args.predictor_variable,
                                    target_variable=args.target_variable, config=config, model=model,
                                    node_dict=node_dict)

    # Write report and include timestamp in the file name to ensure uniqueness
    index_of_extension = args.report_path.rfind('.')
    identifier = str(int(time.time()))
    file_path = args.report_path[:index_of_extension] + identifier + args.report_path[index_of_extension:]

    with open(file=file_path, mode='w') as f:
        f.write(argument_notice)
        f.write('\n\nResults:\n')
        f.write(f'MSE = {mse}\nAccuracy = {acc}\n\nConfusion Matrix: \n{cm}')


def predict_and_test(data_path: str, predictor_variable: str, target_variable: str, config: dict, model: Word2Vec,
                     node_dict: dict) -> tuple[float, float, str]:
    """
    Performs prediction on test set and writes report which demonstrate the fit of the model.

    :param data_path:           path to the data source
    :param predictor_variable:  node type whose similarity to each target should be predicted
    :param target_variable:     node type which forms the possible ratings
    :param config:              dictionary containing graph configuration
    :param model:               trained Word2Vec model
    :param node_dict:           dictionary mapping original ids to unique ids
    :return:                    accuracy, mean squared error and confusion matrix of prediction results
    """

    logger.trace(f'predict_and_test({data_path}, {predictor_variable}, {target_variable}, {config}, {model}, {node_dict})')

    df = pd.read_csv(filepath_or_buffer=data_path,
                     sep=config['data']['separator'],
                     encoding=get_encoding(file=data_path))

    y_prediction, target_column, suffix = predict_from_data(config=config, df=df, model=model, node_dict=node_dict,
                                                            predictor_variable=predictor_variable,
                                                            target_variable=target_variable)

    y_true = [int(y) for y in df[target_column].to_list()]

    # Compute prediction results
    mse = mean_squared_error(y_true=y_true, y_pred=y_prediction)
    acc = accuracy_score(y_true=y_true, y_pred=y_prediction)
    cm = confusion_matrix(y_true=y_true, y_pred=y_prediction, labels=[int(i) for i in suffix])
    return acc, cm, mse


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict data')
    parser.add_argument('-dp', '--data-path', default='./data/test_user_ratings.csv', type=str, help='Path to test data')
    parser.add_argument('-mp', '--model-path', default='./models/word2vec.obj', type=str, help='Path to rec2vec model')
    parser.add_argument('-ndp', '--node-dict-path', default='./output/node_dict.obj', type=str, help='Path to nodedict')
    parser.add_argument('-rp', '--report-path', default='./output/report.txt', type=str, help='Path to report')
    parser.add_argument('-t', '--target-variable', default='ratings:rating', type=str, help='Link to be predicted')
    parser.add_argument('-p', '--predictor-variable', default='users:userID;movies:movieID', type=str, help='Predictor nodes')
    parser.add_argument('-cp', '--config-path', default='./rec2vec/configs/graph_config.yaml', type=str, help='Path to custom config')
    args = parser.parse_args()
    _report_prediction(args=args)


if __name__ == '__main__':
    exit(main())