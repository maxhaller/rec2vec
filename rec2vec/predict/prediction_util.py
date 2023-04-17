import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from rec2vec.predict import prediction_data_loader
from rec2vec import logger


def predict(model: Word2Vec, predictor: str, target_seq: list[str]) -> int:
    """
    Predicts the best fitting node for a predictor, given a target sequence.
    A predictor could be the user id and a target sequence could be a list of possible ratings for an item.

    This method assumes that IDs from the raw data have been converted already and are ordered by their original suffix.
    E.g.    original data: [m_932_1, m_932_2, m_932_3]
            provided data: [421, 422, 423]
            With: m_932_1 => 421, m_932_2 => 422, m_932_3 => 423

    This way, the Word2Vec model knows all the datapoints and returns the index of the rating
    with the maximum similarity.

    :param model:       trained model
    :param predictor:   predictor variable
    :param target_seq:  list of possible targets that are compared to each other
    :return:            index of target with maximum similarity
    """

    logger.trace(f'predict_variable({model}, {predictor}, {target_seq})')

    # Initialization, those values should not be returned and have to be overriden
    max_similarity = float('-inf')
    result = -1

    # For each target, compute its similarity to the predictor
    for i, target in enumerate(target_seq):
        similarity = model.wv.similarity(predictor, target)
        if similarity > max_similarity:
            max_similarity = similarity
            result = int(i)

    return result


def predict_from_data(config: dict, df: pd.DataFrame, model: Word2Vec, node_dict: dict,
                      predictor_variable: str, target_variable: str) -> tuple[list[int], str, list[str]]:
    """
    Computes most similar node of the target (usually an extension) to the predictor node. Usually, the target
    is a rating which extends an item, meaning it is a numeric range encoded as node for each node representing an item.
    The predictor could then be user ids. The prediction then means that the user most probably rates the target
    with a rating that matches the extension.

    :param config:              dictionary containing graph configuration
    :param df:                  data frame containing columns of interest
    :param model:               trained Word2Vec model
    :param node_dict:           dictionary mapping original ids to unique ids
    :param predictor_variable:  node type whose similarity to each target should be predicted
    :param target_variable:     node type which forms the possible ratings
    :return:                    predictions, transformed target values, suffix for extended target nodes
    """

    logger.trace(f'predict_from_data({config}, {df}, {model}, {node_dict}, {predictor_variable}, {target_variable})')

    # Parse user input to get node types and relevant columns for predictions
    predictor_columns_list, suffix, target_column, target_node_type, target_prefix = \
        prediction_data_loader.get_predictor_column_list(config=config, df=df,
                                                         predictor_variable=predictor_variable,
                                                         target_variable=target_variable)

    # Construct x and y for prediction
    # x... unique IDs of predictor variable
    # y... unique IDs of extended target variable
    predictor_list, target_list = prediction_data_loader.create_x_and_y(config, node_dict, predictor_columns_list,
                                                                        suffix, target_node_type, target_prefix)

    # data_rows = [[target1_1, target1_2, target1_3, ...], [predictor1, predictor2, ...]]
    data_rows = zip(target_list, predictor_list)
    y_prediction = [predict(model=model,
                            target_seq=target,
                            predictor=predictor) for target, predictor in tqdm(data_rows)]

    return y_prediction, target_column, suffix
