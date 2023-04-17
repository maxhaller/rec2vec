from rec2vec.util.load_config import load_config
from os.path import exists
from pickle import load, dump
from tqdm import tqdm
from rec2vec import logger
from rec2vec.util.encoding_detector import get_encoding

import pandas as pd


def _extend_id(original_id: str, extension_range: str) -> list[str]:
    """
    Returns the extended id for nodes that extend a node type.
    E.g. rating extends movie:
    m_932 -> [m_932_1, m_932_2, m_932_3, m_932_4, m_932_5]

    :param original_id:         id of extended node
    :param extension_range:     range of extension (e.g. 1-5)
    :return:                    returns list of generated ids
    """

    logger.trace(f'_extend_id({original_id}, {extension_range})')

    lower_bound = int(extension_range.split('-')[0])
    upper_bound = int(extension_range.split('-')[1]) + 1  # range(x, y) -> y is exclusive
    return [f'{original_id}_{str(i)}' for i in range(lower_bound, upper_bound)]


def _return_dict_if_exists(path: str) -> dict | None:
    """
    Returns a stored object if it exists.

    :param path:    path to the (potentially stored) object (= file)
    :return:        stored object or None if that object could not be found
    """

    logger.trace(f'_return_dict_if_exists({path})')

    if exists(path):
        filehandler = open(file=path, mode='rb')
        stored_object = load(filehandler)
        filehandler.close()
        return stored_object
    return None


def _store_and_return_dict(obj: dict, path: str) -> dict:
    """
    Stores and returns object.

    :param obj:     dictionary to be stored
    :param path:    path to the place where the dictionary should be stored
    :return:        dictionary
    """

    logger.trace(f'_store_and_return_dict({obj}, {path})')

    filehandler = open(file=path, mode='wb')
    dump(obj=obj, file=filehandler)
    filehandler.close()
    return obj


def _remove_zero_decimal_place(s: str) -> str:
    """
    Removes the last two characters if string ends with sequence '.0'.
    E.g. '75.0' -> '75'

    :param s:   any string (usually a numeric ID)
    :return:    string without '.0' at the end
    """

    logger.trace(f'_remove_zero_decimal_place({s})')

    return s[:-2] if s.endswith('.0') else s


def _generate_id(value: str, node: dict) -> str:
    """
    Generates an ID for the node dictionary. If configured, a prefix is added to the
    original ID to distinguish nodes of different types from each other.
    E.g. movie node 932 -> m_932

    If no prefix is configured, the ID stays the same.

    :param value:   original ID
    :param node:    corresponding node in configuration (e.g. nodes.movies)
    :return:        generated ID
    """

    logger.trace(f'_generate_id({value}, {node})')

    v = _remove_zero_decimal_place(value)
    return f'{node["id_prefix"]}_{v}' if 'id_prefix' in node else v


def load_nodes(config: dict = None) -> dict[str, dict[str, str]]:
    """
    Loads all nodes from csv files. Node types, data sources and target columns
    have to be specified in the corresponding graph_config.yaml file.

    The final dictionary maps generated IDs to a unique ID.
    original_ids_dict = { 'm_932': 1, 'm_1238': 2 }

    This way, the original (but potentially processed) IDs (that are needed for
    constructing the graph) point to a unique ID that identifies a certain node.
    This is helpful because the rest of the application can use unique IDs.

    :param config:
    :return:
    """

    logger.trace(f'load_nodes({config})')

    if config is None:
        config = load_config()

    logger.debug(f'load_nodes({config})')

    output_folder = config['data']['output_folder']
    node_dict_location = output_folder + config['data']['objects']['node_dict']

    # If graph has been constructed once already, just load stored graph
    stored_object = _return_dict_if_exists(node_dict_location)
    if stored_object is not None:
        logger.debug(f'returning stored object from {node_dict_location}')
        return stored_object
    logger.debug(f'no object found at {node_dict_location}')

    original_ids_dict = {}
    id_counter = 0  # counter for to ensure uniqueness of IDs

    logger.info('extracting nodes...')

    # For every configured node...
    for node in tqdm(config['nodes']):
        node_name = node
        node = config['nodes'][node]

        # Add an entry to the dictionary
        original_ids_dict[node_name] = {}

        # Read the data source which contains the data of that node type line by line...
        filepath = config['data']['folder'] + node['source']
        df = pd.read_csv(filepath_or_buffer=filepath,
                         sep=config['data']['separator'],
                         encoding=get_encoding(file=filepath))
        for _, row in df.iterrows():
            # Generate an ID that contains the prefix of a node and add it to the dictionary
            # <prefix_id, [unique_id]> | <m_932, 1>
            generated_id = _generate_id(value=str(row[node['column']]), node=node)
            original_ids_dict[node_name][generated_id] = id_counter

            # If a node is extended by another node...
            if 'extended' in node:

                # Add the extending node type to the dictionary if it does not already exist
                if node['extended']['by'] not in original_ids_dict:
                    original_ids_dict[node['extended']['by']] = {}

                # Generate a list of ids for the extending node type and store nodes in dictionary
                # E.g. [m_932_1, m_932_2, m_932_3, m_932_4, m_932_5]
                for extended_id in _extend_id(original_id=generated_id, extension_range=node['extended']['range']):
                    id_counter += 1
                    original_ids_dict[node['extended']['by']][extended_id] = id_counter
            id_counter += 1

    logger.info('storing extracted nodes...')
    return _store_and_return_dict(obj=original_ids_dict, path=node_dict_location)


def _get_prefix(edge: dict, vertex: str, config: dict) -> str:
    """
    Returns prefix of a certain vertex in an edge.

    :param edge:    an edge that connects two vertices
    :param vertex:  a string specifying which vertex is under observation (depending on configuration)
    :param config:  configuration that specifies prefixes
    :return:        id prefix for a vertex type (E.g. 'm_') or empty string if no prefix defined
    """

    logger.trace(f'_get_prefix({edge}, {vertex}, {config})')

    # If the node is extending another node, use the id_prefix of the extended node type
    # E.g. a rating has id_prefix m_ because it extends movies
    if 'extending' in edge[f'vertex{vertex}']:
        if 'id_prefix' in config['nodes'][edge[f'vertex{vertex}']['extending']]:
            return config['nodes'][edge[f'vertex{vertex}']['extending']]['id_prefix'] + '_'
        
    # If a node has an id_prefix, simply look it up and return it (E.g. m_)
    if 'id_prefix' in config['nodes'][edge[f'vertex{vertex}']['type']]:
        return config['nodes'][edge[f'vertex{vertex}']['type']]['id_prefix'] + '_'
    
    # No prefix defined -> empty string
    else:
        return ''


def _get_vertex_value_and_type(edge: dict, config: dict, row: pd.Series, vertex: str) -> tuple[str, str]:
    """
    Returns the generated ID and the type of node referenced in the row.

    :param edge:    an edge that connects vertices
    :param config:  configuration that stores columns of interest
    :param row:     row containing columns of interest
    :param vertex:  a string specifying which vertex is under observation (depending on configuration)
    :return:        ID and type of vertex referenced in a row
    """

    logger.trace(f'_get_vertex_value_and_type({edge}, {config}, {row}, {vertex})')

    if 'extend_with' not in edge[f'vertex{vertex}']:
        value = _remove_zero_decimal_place(str(row[edge[f'vertex{vertex}']['column']]))
        return _get_prefix(edge=edge, vertex=vertex, config=config) + value,  \
            edge[f'vertex{vertex}']['type']
    else:
        extension = _remove_zero_decimal_place(str(round(int(row[edge[f'vertex{vertex}']['extend_with']]))))
        value = _remove_zero_decimal_place(str(row[edge[f'vertex{vertex}']['column']]))
        return _get_prefix(edge=edge, vertex=vertex, config=config) + value + f'_{extension}', \
            edge[f'vertex{vertex}']['type']


def load_edges(config: dict = None) -> dict[str, list[str]]:
    """
    Loads edges, meaning it connects vertices that are created from the function load_nodes().
    The combination of edges and vertices form a unidirectional graph.

    :param config:  configuration that defines edges between vertices
    :return:        dictionary <id, [neighbor1.id, neighbor2.id, ...]
    """

    logger.trace(f'load_edges({config})')

    if config is None:
        config = load_config()

    logger.debug(f'load_edges({config})')

    graph = {}
    original_ids_dict = load_nodes(config=config)
    data_folder = config['data']['folder']
    output_folder = config['data']['output_folder']
    separator = config['data']['separator']
    graph_filename = config['data']['objects']['final_graph']
    graph_location = output_folder + graph_filename

    # If graph has been constructed once already, just load stored graph
    stored_object = _return_dict_if_exists(graph_location)
    if stored_object is not None:
        logger.debug(f'returning stored object from {graph_location}')
        return stored_object

    logger.debug(f'no object found at {graph_location}')
    logger.info('extracting edges...')

    # For every edge...
    for edge in tqdm(config['edges']):
        edge = config['edges'][edge]
        filepath = str(data_folder) + str(edge['source'])

        # Read the source file that contains rows that connect vertices line by line...
        df = pd.read_csv(filepath_or_buffer=filepath, sep=separator, encoding=get_encoding(file=filepath))
        for _, row in df.iterrows():
            v1_value, v1_type = _get_vertex_value_and_type(edge=edge, config=config, row=row, vertex='1')
            v2_value, v2_type = _get_vertex_value_and_type(edge=edge, config=config, row=row, vertex='2')

            # look up unique ID using the generated IDs
            unique_v1_value = original_ids_dict[v1_type][v1_value]
            unique_v2_value = original_ids_dict[v2_type][v2_value]

            # Init neighbor lists in case a node does not yet exist in the graph
            if unique_v1_value not in graph:
                graph[unique_v1_value] = []
            if unique_v2_value not in graph:
                graph[unique_v2_value] = []

            # Add unique ID of v2 to the neighbors of v1 (unidirectional!)
            graph[unique_v1_value].append(unique_v2_value)

    logger.info('storing edges...')
    return _store_and_return_dict(obj=graph, path=graph_location)
