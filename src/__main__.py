"""
Main script, only used to call the specified pipeline from arguments.
"""
import argparse

from src.tools import utils, exceptions
from src.tools.startup import params, logger


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-n", "--pipeline_name", type=str, help="pipeline name")
    args = argParser.parse_args()
    pipeline_name = args.pipeline_name

    try:
        pipeline = utils.import_pipeline(pipeline_name)
    except ModuleNotFoundError:
        raise exceptions.PipelineDoesNotExists(pipeline_name)

    logger.info(f"Starting pipeline '{pipeline_name}'")
    # Generate empty settings for global and/or pipeline if needed
    for param_type in ['global', pipeline_name]:
        if param_type not in params:
            params[param_type] = {}
    result = pipeline.execute(params[pipeline_name], params['global'])
    logger.info('Pipeline finished.')

    if result:
        logger.info(f'Result: {result}')
