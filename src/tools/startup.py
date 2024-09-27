import datetime
from envyaml import EnvYAML
import logging
import os
import sys
import yaml

from src.tools import utils


# Load settings
def load_parameters(
        main_params_path: str = 'config/main.yaml',
        internal_params_path: str = 'config/settings.yaml') -> dict:
    """
    Read parameters and load them, both main ones and internal ones.

    Args:
        main_params_path: Path of file containing main parameters.
        internal_params_path: Path of file containing internal parameters.

    Returns:
        Object with all parameters.
    """
    os.environ['EXEC_NAME'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    internal_params = EnvYAML(internal_params_path).export()
    with open(main_params_path, encoding='utf8') as par_file:
        main_params = yaml.safe_load(par_file)
    params = utils.perform_dict_union_recursively(main_params, internal_params)
    return params


# Set logger
params = load_parameters()
log_params = params['global']['logging']
logger = logging.getLogger(__name__)
hdlr_out = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(**log_params['formatter'])
hdlr_out.setFormatter(formatter)
logger.addHandler(hdlr_out)
if log_params['file']:
    hdlr_file = logging.FileHandler(log_params['file'])
    hdlr_file.setFormatter(formatter)
    logger.addHandler(hdlr_file)
logger.setLevel(getattr(logging, log_params['level']))
logger.propagate = False

logger.info('Logger initialized')
