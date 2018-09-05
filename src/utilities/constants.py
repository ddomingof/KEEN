# -*- coding: utf-8 -*-
# Basic Directory
from utilities.config_utils import create_base_dir

KG_EMBEDDINGS_PIPELINE_DIR = create_base_dir()

# Related to corpus Walking RDF and OWL
WROC = 'walking_rdf_and_owl_corpus'
WROC_URL = 'http://aber-owl.net/aber-owl/bio2vec/bio-knowledge-graph.n3'

# WN18 related
WN18_URL = 'https://github.com/Mrlyk423/Relation_Extraction/raw/master/data.zip'
WN_18 = 'wn_18'
WN_18_TRAIN = 'train'
WN_18_VALID = 'valid'
WN_18_TEST = 'test'

# Related to CSQA filtered Wikidata
CSQA_WIKIDATA = 'csqa_wiki_data'

# Configuration related
CLASS_NAME = 'class_name'
## Reader
READER = 'reader'
WROC_READER = 'WROCReader'
CSQA_WIKIDATA_READER = 'CSQAWikiDataReader'
WN18_READER = 'WN18Reader'
## KG embedding model
KG_EMBEDDING_MODEL = 'kg_embedding_model'
# Evaluator
EVALUATOR = 'evaluator'
MEAN_RANK_EVALUATOR = 'MeanRankEvaluator'
# Output paths
ENTITY_TO_EMBEDDINGS = 'entity_to_embeddings'
EVAL_RESULTS = 'eval_results'
# Device related
PREFERRED_DEVICE = 'preferred_device'
CPU = 'cpu'
GPU = 'gpu'
RANDOM_SEARCH_OPTIMIZER = 'random_search_optimizer'

## Metrics
MEAN_RANK = 'mean_rank'
HITS_AT_K = 'hits@k'
# ML params
BATCH_SIZE = 'batch_size'
VOCAB_SIZE = 'vocab_size'
EMBEDDING_DIM = 'embedding_dim'
MARGIN_LOSS = 'margin_loss'
NUM_ENTITIES = 'num_entities'
NUM_RELATIONS = 'num_relations'
NUM_EPOCHS = 'num_epochs'
NUM_OF_MAX_HPO_ITERS = 'maximum_number_of_hpo_iters'
LEARNING_RATE = 'learning_rate'
CONV_E_INPUT_DROPOUT = 'conv_e_input_dropout'
CONV_E_OUTPUT_DROPOUT = 'conv_e_output_dropout'
CONV_E_FEATURE_MAP_DROPOUT = 'conv_e_feature_map_dropout'
EVAL_METRICS = 'eval_metrics'
K_FOR_HITS_AT_K = 'k_for_hits_at_k'

# Further Constants
SEED = 'seed'

# Command line interface parameters
EMBEDDING_DIMENSION_PRINT_MSG = 'Please type the range of preferred embedding dimensions comma separated (e.g. 50,100,200):'
EMBEDDING_DIMENSION_PROMPT_MSG = '> Please select the embedding dimensions:'
EMBEDDING_DIMENSION_ERROR_MSG = 'Invalid input, please positice integer as embedding dimensions.'

BATCH_SIZES_PRINT_MSG = 'Please type the range of preferred batch sizes comma separated (e.g. 32, 64, 128):'
BATCH_SIZES_PROMPT_MSG = '> Please select the batch size(s):'
BATCH_SIZES_ERROR_MSG = 'Invalid input, please select integers as batch size(s)'

EPOCHS_PRINT_MSG = 'Please type the range of preferred epochs comma separated (e.g. 1, 5, 100):'
EPOCHS_PROMPT_MSG = '> Epochs:'
EPOCHS_ERROR_MSG = 'Invalid input, please select integers as epoch(s)'

LEARNING_RATES_PRINT_MSG = 'Please type the range of preferred learning rate(s) comma separated (e.g. 0.1, 0.01, 0.0001:'
LEARNING_RATES_PROMPT_MSG = '> Please select the learning rate(s):'
LEARNING_RATES_ERROR_MSG = 'Invalid input, please float values for the learning rate(s).'

MARGIN_LOSSES_PRINT_MSG = 'Please type the range of preferred margin losse(s) comma separated  (e.g. 1,2,10):'
MARGIN_LOSSES_PROMPT_MSG = '> Please select the margin losse(s):'
MARGIN_LOSSES_ERROR_MSG = 'Invalid input, please type in positive integer as embedding dimensions.'

MAX_HPO_ITERS_PRINT_MSG = 'Please type the maximum number of iterationns for they hyper-parameter search:'
MAX_HPO_ITERS_PROMPT_MSG = '> Maximum number of iterations: '
MAX_HPO_ITERS_ERROR_MSG = 'Invalid input, please type in a positive integer for the maximum number of iterations.'

EMBEDDING_DIMENSION_PRINT_MSG = 'Please type the preferred embedding dimension:'
EMBEDDING_DIMENSION_PROMPT_MSG = '> Please select the embedding dimension:'
EMBEDDING_DIMENSION_ERROR_MSG = 'Invalid input, please type in integer as embedding dimension.'

MARGIN_LOSS_PRINT_MSG = 'Please type in the margin losses:'
MARGIN_LOSS_PROMPT_MSG = '> Margin loss:'
MARGIN_LOSS_ERROR_MSG = 'Invalid input, please type in a float value.'

LEARNING_RATE_PRINT_MSG = 'Please type in the learning rate:'
LEARNING_RATE_PROMPT_MSG = '> Learning rate:'
LEARNING_RATE_ERROR_MSG = 'Invalid input, please type in a float value.'

BATCH_SIZE_PRINT_MSG = 'Please type the batch size comma:'
BATCH_SIZE_PROMPT_MSG = '> Batch size:'
BATCH_SIZE_ERROR_MSG = 'Invalid input, please select a integer.'

EPOCH_PRINT_MSG = 'Please type the number of epochs:'
EPOCH_PROMPT_MSG = '> Epochs:'
EPOCH_ERROR_MSG = 'Invalid input, please select an integers.'

SAVE_CONFIG_PRINT_MSG = 'Do you want to save the configuration file?'
SAVE_CONFIG_PROMPT_MSG = '> \'yes\' or \'no\':'
SAVE_CONFIG_ERROR_MSG = 'Invalid input, please type \'yes\' or \'no\'.'

K_FOR_HITS_AT_K_PRINT_MSG = 'Please select \'k\' for hits@k'
K_FOR_HITS_AT_K_PROMPT_MSG = '> k:'
K_FOR_HITS_AT_K_ERROR_MSG = 'Invalid input \'k\' must be a positive integer.'

CONV_E_INPUT_CHANNELS_PRINT_MSG = 'Please select (comma seperated) the number of input channels for ConvE'
CONV_E_INPUT_CHANNELS_PROMPT_MSG = '> Input channels:'
CONV_E_INPUT_CHANNELS_ERROR_MSG = 'Invalid input, input channels must be positive integers.'

CONV_E_OUT_CHANNELS_PRINT_MSG = 'Please select (comma seperated) the number of output channels for ConvE'
CONV_E_OUT_CHANNELS_PROMPT_MSG = '> Output channels:'
CONV_E_OUT_CHANNELS_ERROR_MSG = 'Invalid input, output channels must be positive integers.'

CONV_E_KERNEL_HEIGHTS_PRINT_MSG = 'Please select the kernel heights for ConvE'
CONV_E_KERNEL_HEIGHTS_PROMPT_MSG = '> Kernel height for defined height %d:'
CONV_E_KERNEL_HEIGHTS_ERROR_MSG = 'Invalid input, kernel heights must be positive integers and <= than %d (defined height).'

CONV_E_KERNEL_WIDTHS_PRINT_MSG = 'Please select the kernel widths for ConvE'
CONV_E_KERNEL_WIDTHS_PROMPT_MSG = '> Kernel width for defined width %d:'
CONV_E_KERNEL_WIDTHS_ERROR_MSG = 'Invalid input, kernel widths mus be positive integers and <= than %d (defined width).'

TRAINING_SET_PRINT_MSG = 'Please provide the path to the training file.'
VALIDATION_SET_PRINT_MSG = 'Please provide the path to the validation file.'
CONFIG_FILE_PRINT_MSG = 'Please provide the path to the configuration file.'

CONV_E_INPUT_DROPOUT_PRINT_MSG = 'Please select (comma separated) the input dropout value(s)'
CONV_E_INPUT_DROPOUT_PROMPT_MSG = '> Input dropout value(s):'
CONV_E_INPUT_DROPOUT_ERROR_MSG = 'Invalid input, input must be positive float values.'

CONV_E_OUTPUT_DROPOUT_PRINT_MSG = 'Please select (comma separated) the output dropout value(s)'
CONV_E_OUTPUT_DROPOUT_PROMPT_MSG = '> Output dropout value(s):'
CONV_E_OUTPUT_DROPOUT_ERROR_MSG = 'Invalid output, input must be positive float values.'

CONV_E_FEATURE_MAP_DROPOUT_PRINT_MSG = 'Please select (comma separated) the feature map dropout value(s)'
CONV_E_FEATURE_MAP_DROPOUT_PROMPT_MSG = '> Feature map dropout value(s):'
CONV_E_FEATURE_MAP_DROPOUT_ERROR_MSG = 'Invalid output, input must be positive float values.'

TRAINING = 'training'
HYPER_PARAMTER_SEARCH = 'hyper_parameter_search'
HYPER_PARAMTER_OPTIMIZATION_PARAMS = 'hyper_optimization_params'
TRAINING_SET_PATH = 'training_set_path'
VALIDATION_SET_PATH = 'validation_set_path'
VALIDATION_SET_RATIO = 'validation_set_ratio'
NORMALIZATION_OF_ENTITIES = 'normalization_of_entities'

# Model names
CONV_E = 'ConvE'
TRANS_E = 'TransE'
TRANS_H = 'TransH'
TRANS_D = 'TransD'
TRANS_R = 'TransR'
RESCAL = 'RESCAL'

CONV_E_HEIGHT = 'ConvE_height'
CONV_E_WIDTH = 'ConvE_width'
CONV_E_INPUT_CHANNELS = 'ConvE_input_channels'
CONV_E_OUTPUT_CHANNELS = 'ConvE_output_channels'
CONV_E_KERNEL_HEIGHT = 'ConvE_kernel_heights'
CONV_E_KERNEL_WIDTH = 'ConvE_kernel_widths'

OUTPUT_DIREC = 'output_direc'
