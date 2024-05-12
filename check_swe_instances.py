import json
import logging
import os
import re
import traceback
import yaml

from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from rich.logging import RichHandler
from simple_parsing import parse
from simple_parsing.helpers import Serializable, FrozenSerializable, FlattenedAccess
from sweagent import (
    Agent,
    AgentArguments,
    EnvironmentArguments,
    ModelArguments,
    SWEEnv,
    get_data_path_name,
)
from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from unidiff import PatchSet

handler = RichHandler(show_time=False, show_path=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger("run_dev")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
logging.getLogger("simple_parsing").setLevel(logging.WARNING)

from run import ScriptArguments, save_arguments, should_skip, save_predictions, main, ActionsArguments

from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # Import process_map from tqdm.contrib.concurrent


import os
import sys

#####################################################################
# model inputs
#####################################################################

model_name = "azure:gpt4"
per_instance_cost = 2. 
suffix = "dummy_run_1"
split = "dev"
split = "test"

data_path = "princeton-nlp/SWE-bench_Lite"
config_file = "config/default.yaml"
config_file = "config/default_epllm-v0.1.yaml"
config_file = "config/default_hepllm_v0.1.yaml"
config_file = "config/hepllm/default-v1-root-level.yaml"
# config_file = "config/hepllm/default-v1-leaf-level.yaml"


# data_path = "https://github.com/pvlib/pvlib-python/issues/1603"
# config_file = "config/default_from_url.yaml"
# config_file = "config/default_from_url_epllm-v0.1.yaml"


defaults = ScriptArguments(
        suffix=suffix,
        num_tasks = 1, # useful for debugging on the swe-bench-lite dataset
        environment=EnvironmentArguments(
            image_name="ghcr.io/xingyaoww/eval-swe-bench-all:lite-v1.1",#"swe-agent",
            data_path=data_path,
            split=split,
            verbose=True,
            install_environment=True,
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name=model_name,
                total_cost_limit=0.0,
                per_instance_cost_limit=2.0,
                temperature=0.2,
                top_p=0.95,
            ),
            config_file=config_file,
        ),
    actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
    use_dockerized_inference=True,
    run_and_eval=False,
    )

args = defaults

#####################################################################

def check_env_index(index):
    # sys.stdout = open(os.devnull, 'w')  # Redirect stdout to nowhere
    # sys.stderr = open(os.devnull, 'w')  # Redirect stderr to nowhere
    env_ = SWEEnv(args.environment, use_dockerized_inference=True)
    try:
        env_.reset_alternative(index, perform_sanity_check=True)
        env_.close()
        logger.info(f"Index {index} is OK ✅.")
        return index, True
    except Exception as e:
        # return the exception as a string instead of True
        env_.close()
        logger.info(f"Index {index} is NOT OK ❌ ... due to error: {str(e)}.")
        return index, str(e)
    
if __name__ == "__main__":
    indices = range(300)
    # Use process_map instead of Pool.map directly
    results = process_map(check_env_index, indices, max_workers=8, chunksize=1)

    # Convert the results to a dictionary
    results_dict = dict(results)

    # compile the results and print total OK (i.e True) and NOT OK (i.e. string error)
    output_results = {
        'total_ok': sum([1 for k, v in results_dict.items() if v == True]),
        'total_not_ok': sum([1 for k, v in results_dict.items() if v != True]),
        'indices_not_ok': [(k, v) for k, v in results_dict.items() if v != True],
        'indices_ok': [(k, v) for k, v in results_dict.items() if v == True],
        'output_results': results_dict
    }
    
    logger.info("##############################################")
    logger.info(f"Total OK: {output_results['total_ok']}")
    logger.info(f"Total NOT OK: {output_results['total_not_ok']}")
    logger.info(f"Indices NOT OK: {output_results['indices_not_ok']}")
    logger.info("##############################################")

    # save the results to a file
    with open(f'swe_instances_sanity_results.json', 'w') as f:
        json.dump(output_results, f, indent=4)

    # also print the results
    print(output_results)



