from rec2vec import logger

import pandas as pd


def get_node_prefix(config: dict, node_type: str) -> str:
    """
    Extracts a nodes prefix from the config. The returned prefix already contains an underscore.
    E.g. movies -> m_

    :param config:      dictionary, containing the graph configuration
    :param node_type:   node type of interest
    :return:            prefix of this node type
    """

    logger.trace(f'get_node_prefix({config}, {node_type})')

    return config['nodes'][node_type]['id_prefix'] + '_' if 'id_prefix' in config['nodes'][node_type] else ''


def extract_suffix_and_prefix_from_extended_node(config: dict, target_node_type: str, node_type: str) \
        -> tuple[list[str], str]:
    """
    Extracts the suffix (list of the range of extensions) and the prefix of the target node type.

    :param config:              configuration of the graph (dictionary)
    :param target_node_type:    node type of the target variable
    :param node_type:           node type which is extended by the target variable
    :return:
    """

    logger.trace(f'extract_suffix_and_prefix_from_extended_node({config}, {target_node_type}, {node_type})')

    suffix, target_prefix = None, None

    # Find node that is extended by the target node
    if 'extended' in config['nodes'][node_type]:
        if config['nodes'][node_type]['extended']['by'] == target_node_type:
            target_prefix = config['nodes'][node_type]['id_prefix']

        # Parse suffix range and put values in a list
        range_string = config['nodes'][node_type]['extended']['range']
        lower_bound = int(range_string.split('-')[0])
        upper_bound = int(range_string.split('-')[1]) + 1
        suffix = [str(i) for i in range(lower_bound, upper_bound)]

    return suffix, target_prefix


def create_x_and_y(config: dict, node_dict: dict, predictor_columns_list: list, suffix: list[str],
                   target_node_type: str, target_prefix) -> tuple[list[list[str]], list[list]]:
    """
    Creates lists of data that can be used for predictions.

    :param config:                  dictionary configuring the graph
    :param node_dict:               dictionary mapping original ids to unique ids
    :param predictor_columns_list:  list of columns used for predictions
    :param suffix:                  list of extensions of target node
    :param target_node_type:        type of target node
    :param target_prefix:           list of extended target nodes [[t1.1, t1.2, ...], [t2.1, t2.2, ...]]
    :return:                        lists with same length, for each predictor there x possible targets
    """

    logger.trace(f'create_x_and_y({config}, {node_dict}, {predictor_columns_list}, {suffix}, {target_node_type}, '
                 f'{target_prefix})')

    target_list_id, predictor_list_id = [], []

    # For every variable...
    for column in predictor_columns_list:

        # If it's a target variable, add it to the target list (identify target by prefix)
        if column[0].startswith(target_prefix + '_'):
            for row in column:
                target_list_id.append([node_dict[target_node_type][row + '_' + i] for i in suffix])
            predictor_columns_list.remove(column)

        # If it's a predictor variable, add it to the predictor list
        else:
            for node in config['nodes']:
                if 'id_prefix' in config['nodes'][node]:
                    if column[0].startswith(config['nodes'][node]['id_prefix'] + '_'):
                        for row in column:
                            predictor_list_id.append(node_dict[node][row])

    return predictor_list_id, target_list_id


def get_predictor_column_list(config: dict, df: pd.DataFrame, predictor_variable: str, target_variable: str) \
        -> tuple[list[list[str]], list[str], str, str, str]:
    """
    Extracts columns, suffix and target properties from config and dataframe.
    It is important that the user input has the format:
        'type:column'               for one node
        'type:column;type2:column2' for two nodes

    :param config:              dictionary configuring the graph
    :param df:                  dataframe containing the data
    :param predictor_variable:  node type and ID to be used for predicting the target (nodetype:id)
    :param target_variable:     node type and ID of the nodes that are to be predicted (nodetype:id)
    :return:                    columns, suffix and target properties from config and dataframe
    """

    logger.trace(f'get_predictor_column_list({config}, {df}, {predictor_variable}, {target_variable})')

    # Parse user input for predictor variable
    predictor_input = predictor_variable
    predictor_variables = predictor_input.split(';')
    predictor_list, predictor_columns_list = [], []

    # Potentially support multiple variables in the future, currently just one target can be defined
    for var in predictor_variables:
        predictor_list.append([var.split(':')[0], var.split(':')[1]])

    # Parse user input for target variable
    target_input = target_variable
    target_node_type = target_input.split(':')[0]
    target_column = target_input.split(':')[1]
    suffix = target_prefix = None
    for var in predictor_list:

        # Extract suffix and prefix for the target node (extension of some other node)
        suffix, target_prefix = extract_suffix_and_prefix_from_extended_node(config=config, target_node_type=target_node_type, node_type=var[0])
        prefix = get_node_prefix(config, var[0])

        # Manipulates original IDs by adding prefixes, so that they can be found in the node dictionary (node_dict)
        df[var[1]] = [prefix + str(row) for row in df[var[1]]]

        # Add the new values to the list
        predictor_ids = df[var[1]].to_list()
        predictor_columns_list.append(predictor_ids)

    # Return extracted information of interest, used for predictions
    return predictor_columns_list, suffix, target_column, target_node_type, target_prefix