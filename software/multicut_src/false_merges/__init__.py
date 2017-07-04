from compute_paths_and_features import shortest_paths, path_feature_aggregator
from false_merges_workflow import compute_false_merges, resolve_merges_with_lifted_edges, project_resolved_objects_to_segmentation
from false_merges_workflow import resolve_merges_with_lifted_edges_global

from os import path, remove
import logging
import sys


# # Create the Logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# # Create the Handler for logging data to a file
# logger_handler = logging.FileHandler('python_logging.log')
# logger_handler.setLevel(logging.DEBUG)
#
# # Create a Formatter for formatting the log messages
# # logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#
# # Add the Formatter to the Handler
# logger_handler.setFormatter(logger_formatter)
#
# # Create handler for output to the console
# h_stream = logging.StreamHandler(sys.stdout)
# h_stream.setLevel(logging.DEBUG)
# h_stream.setFormatter(logger_formatter)
#
# # Add the Handlers to the Logger
# logger.addHandler(logger_handler)
# logger.addHandler(h_stream)
#
# logger.info('Configuration of false merges workflow logger complete!')