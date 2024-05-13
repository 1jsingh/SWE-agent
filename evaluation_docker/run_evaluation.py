#!/usr/bin/env python3

"""Run evaluation"""
import argparse
import asyncio
import hashlib
import logging
import os

from swebench import (
    get_eval_report,
    get_logs_eval,
    get_model_report,
    get_resolution_status,
    run_evaluation,
    get_eval_refs,
)

from swebench.harness.constants import (
    INSTALL_FAIL,
)

from swebench_docker.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION, MAP_REPO_TO_TEST_FRAMEWORK,
)
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_instances, get_test_directives

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")
import json
import os
import time
from collections import Counter
from glob import glob
from typing import List
from unidiff import PatchSet
from rich import print


def deterministic_hash(input_string: str, length: int = None):
    input_bytes = input_string.encode('utf-8')
    sha256_hash = hashlib.sha256(input_bytes)
    hex_digest = sha256_hash.hexdigest()
    if length is None:
        return hex_digest
    return hex_digest[:length]


def validate_predictions(predictions_path, tasks_ids):
    # Check that predictions file exists
    if not any([predictions_path.endswith(x) for x in [".json", ".jsonl"]]):
        raise ValueError("Predictions path must be .json or .jsonl file")
    predictions = get_instances(predictions_path)
    not_in_tasks = []
    # Check that predictions are correctly formatted
    for pred in predictions:
        if any([x not in pred for x in [KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION]]):
            raise ValueError(f"Every prediction must have {KEY_INSTANCE_ID}, {KEY_MODEL}, and {KEY_PREDICTION} fields")
        if pred[KEY_INSTANCE_ID] not in tasks_ids:
            not_in_tasks.append(pred[KEY_INSTANCE_ID])
    # Check that instance IDs specified by predictions exist
    if len(not_in_tasks) > 0:
        logger.warning(
            "Predictions for the following instance_ids were not "
            + "found in the tasks file and will not be considered: "
            + ", ".join(not_in_tasks)
        )

def load_json_file(filepath, max_attempts=5, delay=2):
    """Attempt to load a JSON file up to max_attempts with a delay between tries."""
    for attempt in range(max_attempts):
        try:
            with open(filepath, "r") as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Failed to load JSON file on attempt {attempt+1}: {e}")
            time.sleep(delay)  # Wait before trying again
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None  # or raise an exception, depending on how critical this is
    raise Exception("Maximum attempts reached, JSON file could not be loaded")

async def main2(
    predictions_path: str,
    swe_bench_tasks: str,
    namespace: str,
    log_dir: str,
    log_suffix: str = "",
    skip_existing: bool = False,
    timeout: int = 900,
    num_processes: int = -1,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        namespace (str): Docker repository namespace.
        log_dir (str): Path to the directory where logs will be saved.
        log_suffix (str): Suffix to append to log file names.
        skip_existing (bool): Whether to skip evaluations for predictions that already have logs.
        timeout (int): Timeout for each evaluation.
        num_processes (int): Number of processes to run in parallel (-1 = unlimited)

    Raises:
        ValueError: If log_dir is not a directory, testbed is not a directory, or swe_bench_tasks does not exist.
    """
    # Validate arguments
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")

    tasks = list(get_eval_refs(swe_bench_tasks).values())

    # Verify arguments are formatted correctly
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks}
    predictions_path = os.path.abspath(predictions_path)
    validate_predictions(predictions_path, [t[KEY_INSTANCE_ID] for t in tasks])

    predictions = get_instances(predictions_path)

    if len(predictions) == 0:
        logger.info("No predictions to evaluate")
        return

    # Remove predictions that have already been evaluated
    if skip_existing:
        # Skip logs that already exist
        predictions_filtered = []
        for p in predictions:
            log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log"
            if log_suffix:
                log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.{log_suffix}.eval.log"
            log_file = os.path.join(log_dir, log_file_name)
            if not os.path.exists(log_file):
                predictions_filtered.append(p)
        if len(predictions_filtered) == 0:
            logger.info(f"All predictions already exist, skipping")
            return
        else:
            logger.info(
                f"# of predictions to evaluate: {len(predictions_filtered)} " +
                f"({len(predictions) - len(predictions_filtered)} already evaluated)"
            )
            predictions = predictions_filtered
    else:
        logger.info(
            f"# of predictions to evaluate: {len(predictions)}"
        )

    task_instances = []

    # Set the relevant data on task_instances
    for prediction in predictions:
        task = tasks_map[prediction[KEY_INSTANCE_ID]]

        test_type = MAP_REPO_TO_TEST_FRAMEWORK[task["repo"]]
        test_directives = get_test_directives(task)
        test_cmd = f"{test_type} {' '.join(test_directives)}"

        task_instances.append({
            "repo": task["repo"],
            "version": task["version"],
            "base_commit": task["base_commit"],
            KEY_INSTANCE_ID: prediction[KEY_INSTANCE_ID],
            KEY_MODEL: prediction[KEY_MODEL],
            KEY_PREDICTION: prediction[KEY_PREDICTION],
            "test_patch": task["test_patch"],
            "test_directives": test_directives,
            "test_cmd": test_cmd
        })

    task_instances = sorted(task_instances, key=lambda x: x[KEY_INSTANCE_ID])

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(task_instances))
    async with asyncio.TaskGroup() as tg:
        for task_instance in task_instances:
            if task_instance[KEY_PREDICTION]:
                async def run_docker_throttled(*args, **kwargs):
                    async with sem:
                        return await run_docker_evaluation(*args, **kwargs)

                tg.create_task(run_docker_throttled(task_instance, namespace, log_dir, timeout, log_suffix))
            else:
                logger.info(f"[{task_instance[KEY_INSTANCE_ID]}] No prediction found.")


async def main(
    predictions_path: str,
    swe_bench_tasks: str,
    namespace: str,
    log_dir: str,
    log_suffix: str = "",
    skip_existing: bool = False,
    timeout: int = 900,
    num_processes: int = -1,
):
    # Validate arguments
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")

    tasks = list(get_eval_refs(swe_bench_tasks).values())

    # Verify arguments are formatted correctly
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks}
    predictions_path = os.path.abspath(predictions_path)
    validate_predictions(predictions_path, [t[KEY_INSTANCE_ID] for t in tasks])

    predictions = get_instances(predictions_path)

    if len(predictions) == 0:
        logger.info("No predictions to evaluate")
        return

    # Remove predictions that have already been evaluated
    if skip_existing:
        predictions_filtered = []
        for p in predictions:
            log_file_name = f"{p[KEY_INSTANCE_ID]}.{p[KEY_MODEL]}.eval.log"
            if log_suffix:
                log_file_name = f"{log_file_name}.{log_suffix}.eval.log"
            log_file = os.path.join(log_dir, log_file_name)
            if not os.path.exists(log_file):
                predictions_filtered.append(p)
        # log the number of predictions that were already evaluated
        logger.info(
                f"# of predictions to evaluate: {len(predictions_filtered)} " +
                f"({len(predictions) - len(predictions_filtered)} already evaluated)"
            )
        predictions = predictions_filtered
        if len(predictions) == 0:
            logger.info("All predictions already exist, skipping")
            # return

    logger.info(f"# of predictions to evaluate: {len(predictions)}")

    task_instances = []
    for prediction in predictions:
        task = tasks_map[prediction[KEY_INSTANCE_ID]]
        test_type = MAP_REPO_TO_TEST_FRAMEWORK[task["repo"]]
        test_directives = get_test_directives(task)
        test_cmd = f"{test_type} {' '.join(test_directives)}"
        task_instance = {
            "repo": task["repo"],
            "version": task["version"],
            "base_commit": task["base_commit"],
            KEY_INSTANCE_ID: prediction[KEY_INSTANCE_ID],
            KEY_MODEL: prediction[KEY_MODEL],
            KEY_PREDICTION: prediction[KEY_PREDICTION],
            "test_patch": task["test_patch"],
            "test_directives": test_directives,
            "test_cmd": test_cmd
        }
        task_instances.append(task_instance)

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(task_instances))
    tasks_list = []

    for task_instance in task_instances:
        async def run_evaluation(instance):
            async with sem:
                return await run_docker_evaluation(
                    instance, namespace, log_dir, timeout, log_suffix
                )

        if task_instance[KEY_PREDICTION]:
            tasks_list.append(run_evaluation(task_instance))

    if tasks_list:
        await asyncio.gather(*tasks_list)

    if True:
        ##############################################################################
        # Get predictions, define log_dir
        directory = os.path.dirname(predictions_path)
        directory_name = directory.rsplit("/", 1)[-1]
        predictions = [json.loads(l) for l in open(predictions_path, "r").readlines()]
        # log_dir = os.path.join(log_dir, directory_name)
        print(f"Log directory for evaluation run: {log_dir}")
        eval_refs = get_eval_refs(swe_bench_tasks)

        # Iterate through predictions
        scorecards = []
        for p in predictions:
            scorecard = {KEY_INSTANCE_ID: p[KEY_INSTANCE_ID], "statuses": [], "stats": {}}

            # Add trajectory statistics if traj_path exists
            traj_path = os.path.join(directory, f"{p[KEY_INSTANCE_ID]}.traj")
            alt_traj_path = os.path.join(directory, 'trajs', f"{p[KEY_INSTANCE_ID]}.traj")
            
            # if traj_path does not exist, try alt_traj_path
            if not os.path.exists(traj_path) and os.path.exists(alt_traj_path):
                traj_path = alt_traj_path

            if os.path.exists(traj_path):
                # traj_data = json.load(open(traj_path, "r"))
                traj_data = load_json_file(traj_path)
                scorecard["stats"]["traj_num_steps"] = len(traj_data["trajectory"])
                scorecard["stats"]["traj_action_dist"] = dict(
                    Counter(
                        [
                            entry["action"].strip().split()[0]
                            if entry["role"] == "assistant" and "action" in entry and len(entry["action"]) > 0
                            else None
                            for entry in traj_data["history"]
                        ]
                    )
                )
                scorecard["exit_status"] = (
                    traj_data["info"]["exit_status"]
                    if "exit_status" in traj_data["info"]
                    else "n/a"
                )

            # Check that a prediction was generated
            if p[KEY_PREDICTION] is None or p[KEY_PREDICTION].strip() == "":
                scorecard["statuses"].append("not_generated")
                scorecards.append(scorecard)
                continue
            scorecard["statuses"].append("generated")

            # Get log file
            log_path = os.path.join(
                log_dir, f"{p[KEY_INSTANCE_ID]}.{directory_name}.eval.log"
            )
            if not os.path.exists(log_path):
                log_path = glob.glob(os.path.join(log_dir, f"{p[KEY_INSTANCE_ID]}.*.eval.log"))[0]
            
            if not os.path.exists(log_path):
                scorecard["statuses"].append("build_failure")
                scorecards.append(scorecard)
                continue

            # Get evaluation logs
            eval_sm, found = get_logs_eval(log_path)

            # Check that the prediction generated
            if not found:
                scorecards.append(scorecard)
                continue
            scorecard["statuses"].append("applied")

            with open(log_path, "r") as f:
                log_contents = f.read()
                if INSTALL_FAIL in log_contents:
                    scorecard["statuses"].append("install_fail")

            # Get resolution status
            report = get_eval_report(eval_sm, eval_refs[p[KEY_INSTANCE_ID]])
            scorecard["test_results"] = {
                "failure": {
                    "FAIL_TO_PASS": report["FAIL_TO_PASS"]["failure"],
                    "PASS_TO_PASS": report["PASS_TO_PASS"]["failure"],
                },
                "success": {
                    "FAIL_TO_PASS": report["FAIL_TO_PASS"]["success"],
                    "PASS_TO_PASS": report["PASS_TO_PASS"]["success"],
                }
            }
            resolution_status = get_resolution_status(report)
            scorecard["statuses"].append(resolution_status)

            # # print the resolution status, and in particular scorecard["test_results"] not including scorecard["test_results"]["sucess"]["PASS_TO_PASS"]
            # # surrounded with dividers for easy parsing
            # print("===============================================")
            # print(f"[{p[KEY_INSTANCE_ID]}] Resolution status: {resolution_status}")
            # print(f"[{p[KEY_INSTANCE_ID]}] Test results:")
            # # not including scorecard["test_results"]["sucess"]["PASS_TO_PASS"]
            # scorecard_copy = copy.deepcopy(scorecard)
            # del scorecard_copy["test_results"]["success"]["PASS_TO_PASS"]
            # print(json.dumps(scorecard_copy, indent=2))
            # print("===============================================")

            try:
                diff_obj = PatchSet(p[KEY_PREDICTION])
                scorecard["patch_files"] = [
                    x.path
                    for x in diff_obj.modified_files
                    + diff_obj.added_files
                    + diff_obj.removed_files
                ]
                scorecard["patch_lines_add"] = sum([f.added for f in diff_obj])
                scorecard["patch_lines_del"] = sum([f.removed for f in diff_obj])
            except Exception as e:
                print(f"[{p[KEY_INSTANCE_ID]}] Error parsing prediction diff: {e}")
                scorecard["patch_files"] = []
                scorecard["patch_lines_add"] = 0
                scorecard["patch_lines_del"] = 0
            scorecards.append(scorecard)

        # Calculate cumulative results
        get_ids_with_status = lambda x: [
            s[KEY_INSTANCE_ID] for s in scorecards if x in s["statuses"]
        ]
        report = {
            "# Not Generated": len(get_ids_with_status("not_generated")),
            "# Generated": len(get_ids_with_status("generated")),
            "# Applied": len(get_ids_with_status("applied")),
            "# Resolved": len(get_ids_with_status("RESOLVED_FULL")),
            "# Install Fail": len(get_ids_with_status("install_fail")),
        }
        print(f"== Evaluation Report ==\n{report}")

        report_exits = dict(
            Counter([s["exit_status"] if "exit_status" in s else "n/a" for s in scorecards])
        )

        # Save to summary, scorecard json
        path_scorecards = os.path.join(directory, "scorecards.json")
        with open(path_scorecards, "w") as f:
            json.dump(scorecards, fp=f, indent=2)
        print(f"- Wrote per-instance scorecards to {path_scorecards}")

        path_results = os.path.join(directory, "results.json")
        with open(path_results, "w") as f:
            json.dump(
                {
                    "report": report,
                    "report_exits": report_exits,
                    "not_generated": get_ids_with_status("not_generated"),
                    "generated": get_ids_with_status("generated"),
                    "applied": get_ids_with_status("applied"),
                    "resolved": get_ids_with_status("RESOLVED_FULL"),
                    "install_fail": get_ids_with_status("install_fail"),
                },
                fp=f,
                indent=2,
            )
        print(f"- Wrote summary of run to {path_results}")

        # Sanity check against get_model_report
        report = get_model_report(
            directory_name, predictions_path, swe_bench_tasks, log_dir
        )
        print(f"Reference Report:")
        for k, v in report.items():
            print(f"- {k}: {len(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file", required=True)
    parser.add_argument("--log_dir", type=str, help="Path to log directory", required=True)
    parser.add_argument("--swe_bench_tasks", type=str, help="Path to dataset file or HF datasets name", required=True)
    parser.add_argument("--namespace", type=str, help="Docker repository namespace", required=False, default="aorwall")
    parser.add_argument("--log_suffix", type=str, help="(Optional) Suffix to append to log file names", default="")
    parser.add_argument("--skip_existing", action="store_true", help="(Optional) Skip existing logs")
    parser.add_argument("--timeout", type=int, help="(Optional) Timeout in seconds (default: 900)", default=1800)
    parser.add_argument("--num_processes", type=int, help="(Optional) Number of processes to run in parallel (-1 for unlimited)", default=-1)
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
