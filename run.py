import json
import logging
import os
import re
import traceback
from typing import Any, Dict, Optional
import rich.console
import rich.markdown
import rich.panel
import rich.markdown
import yaml
import numpy as np

from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from rich.logging import RichHandler
from simple_parsing import parse
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from simple_parsing.helpers.flatten import FlattenedAccess
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
from sweagent.environment.utils import InvalidGithubURL, get_associated_commit_urls, get_gh_issue_data, parse_gh_issue_url

# NOTE: used for multiprocessing
from multiprocessing import Pool
from functools import partial

handler = RichHandler(show_time=False, show_path=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger("run_dev")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
logging.getLogger("simple_parsing").setLevel(logging.WARNING)


GOLD_TEST = "/home/jas/project/qstar/SWE-agent/trajectories/gold/azure-gpt4__SWE-bench_Lite__test__default-v4-root-level__t-0.00__p-0.95__c-10.00__install-1__test_0_all_run1_gold_patch/results.json"


@dataclass(frozen=True)
class ActionsArguments(FlattenedAccess, FrozenSerializable):
    """Run real-life actions (opening PRs, etc.) if we can solve the issue."""
    open_pr: bool = False  # Open a PR with the patch if we can solve the issue
    # Skip action if there are already commits claiming to fix the issue. Please only
    # set this to False if you are sure the commits are not fixes or if this is your
    # own repository!
    skip_if_commits_reference_issue: bool = True  
    # For PRs: If you want to push the branch to a fork (e.g., because you lack
    # permissions to push to the main repo), set this to the URL of the fork.
    push_gh_repo_url: str = ""

    def __post_init__(self):
        if not self.skip_if_commits_reference_issue and self.push_gh_repo_url:
            raise ValueError(
                "Overriding `skip_if_commits_reference_issue` when you are "
                "pushing to a fork is not supported. You should manually "
                "apply the patch to the forked repository."
            )

@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    """Configure the control flow of the run.py script"""
    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    instance_filter: str = ".*"  # Only run instances that completely match this regex
    skip_existing: bool = True  # Skip instances with existing trajectories
    suffix: str = ""
    # Raise unhandled exceptions during the run (useful for debugging)
    raise_exceptions: bool = False
    # Number of tasks to run (useful for debugging)
    num_tasks: int = -1
    start_index: int = 0 # NOTE: used for debug
    use_gold_patches: bool = False # NOTE: used for debug
    exp_subdir: str = "latest" # NOTE: used for subdir in trajectories directory
    filter_gold_patch_positives: bool = False # NOTE: used for debug
    gold_patch_results_file: str = "gold_patch_results.json" # NOTE: used for debug
    num_processes: int = 1 # NOTE: used for multiprocessing

    @property
    def run_name(self):
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        data_stem = get_data_path_name(self.environment.data_path)
        data_split = self.environment.split if self.environment.split else "null"
        config_stem = Path(self.agent.config_file).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment

        return (
            f"{model_name}__{data_stem}__{data_split}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )

# use multiprocessing to run multiple tasks in parallel if gold patches are used
def save_gold_patch(index, env, traj_dir):
    trajectory = []
    info = {"submission": env.data[index]["patch"]}
    save_predictions(traj_dir, env.data[index]["instance_id"], info)
    save_patch(traj_dir, env.data[index]["instance_id"], info)


def main(args: ScriptArguments, index=None, num_processes=1):
    if num_processes > 1:
        logger.info(f"🚀 Starting process {index} ...")
    logger.info(f"📙 Arguments: {args.dumps_yaml()}")
    agent = Agent("primary", args.agent)
    env = SWEEnv(args.environment)

    # traj_dir = Path("trajectories") / Path(getuser()) / args.run_name
    traj_dir = Path("trajectories") / Path(args.exp_subdir) / args.run_name
    traj_dir.mkdir(parents=True, exist_ok=True)

    if num_processes == 1 or index == 0:
        save_arguments(traj_dir, args)

    # NOTE: num_tasks is the number of tasks to run, if -1, run all tasks
    num_tasks = args.num_tasks if args.num_tasks > 0 else len(env.data)
    num_tasks = min(num_tasks, len(env.data) - args.start_index)
    assert num_tasks > 0, f"num_tasks={num_tasks} must be positive"
    
    if num_processes == 1:
        start_index_process = 0 
        num_tasks_process = num_tasks
    else:
        num_tasks_process = int(np.ceil(num_tasks / num_processes))
        start_index_process = args.start_index + index * num_tasks_process
        num_tasks_process = min(num_tasks_process, num_tasks - index * num_tasks_process)
        logger.info(f"🚀 Process {index} will run {num_tasks_process} tasks {start_index_process} to {start_index_process + num_tasks_process}")

    # filter gold patch positives
    if args.filter_gold_patch_positives:
        # open the results json file
        with open(args.gold_patch_results_file, "r") as f:
            gold_patch_results = json.load(f)
        # get positive instance ids
            resolved_instance_ids = gold_patch_results['resolved']
            logger.info(f"🔍 Found {len(resolved_instance_ids)} resolved instances ...")

    # # NOTE: if use_gold_patches is set to True, the agent will use the gold patches as the submission
    # # used for debugging purposes 
    # # use multiprocessing to run multiple tasks in parallel if gold patches are used
    # if args.use_gold_patches:
    #     with Pool(4) as p:
    #         # Pass the necessary arguments to the function
    #         func = partial(save_gold_patch, env=env, traj_dir=traj_dir)
    #         p.map(func, range(args.start_index, args.start_index + num_tasks))
    #     return

    for index in range(args.start_index + start_index_process, args.start_index + start_index_process + num_tasks_process):
        # index += args.start_index # TODO: remove this line
        try:
            # NOTE: if use_gold_patches is set to True, the agent will use the gold patches as the submission
            # used for debugging purposes 
            if args.use_gold_patches:
                logger.info(f"🔮 Using gold patches as submission ... task {index}")
                trajectory = []
                info = {"submission": env.data[index]["patch"]}
                save_predictions(traj_dir, env.data[index]["instance_id"], info)
                save_patch(traj_dir, env.data[index]["instance_id"], info)
                continue
            
            # filter gold patch negatives
            if args.filter_gold_patch_positives:
                instance_id = env.data[index]["instance_id"]
                if instance_id not in resolved_instance_ids:
                    logger.info(f"❎ Skipping instance {instance_id} with idx:{index} as the gold patch is not working on this instance ...")
                    continue

            # Reset environment
            instance_id = env.data[index]["instance_id"]
            assert isinstance(instance_id, str)  # mypy
            if should_skip(args, traj_dir, instance_id):
                continue
            logger.info("▶️  Beginning task " + str(index))

            # NOTE: env.reset() will be mainly doing the cloning of the repo to the docker container
            # as well as cleaning the repo, setting it to base commit (before the issue), option of applying test path for oracle setting
            # and potentially install test env (though not clear if its the same as the base env or the agent has to install it himself)
            observation, info = env.reset(index)
            if info is None:
                continue

            # Get info, patch information
            issue = getattr(env, "query", None)
            files = []
            if "patch" in env.record:
                files = "\n".join(
                    [f"- {x.path}" for x in PatchSet(env.record["patch"]).modified_files]
                )
            # Get test files, F2P tests information
            test_files = []
            if "test_patch" in env.record:
                test_patch_obj = PatchSet(env.record["test_patch"])
                test_files = "\n".join(
                    [f"- {x.path}" for x in test_patch_obj.modified_files + test_patch_obj.added_files]
                )
            tests = ""
            if "FAIL_TO_PASS" in env.record:
                tests = "\n".join([f"- {x}" for x in env.record["FAIL_TO_PASS"]])

            setup_args = {
                "issue": issue,
                "files": files,
                "test_files": test_files,
                "tests": tests
            }
            
            
            if args.agent.use_hepllm:
                # info, trajectory = agent.run_hepllm(
                #     setup_args=setup_args,
                #     env=env,
                #     observation=observation,
                #     traj_dir=traj_dir,
                #     return_type="info_trajectory",
                # )
                info, trajectory = agent.run_heirarchial(
                    setup_args=setup_args,
                    env=env,
                    observation=observation,
                    traj_dir=traj_dir,
                    return_type="info_trajectory",
                )
            else:
                # Run the agent (old version)
                info, trajectory = agent.run(
                    setup_args=setup_args,
                    env=env,
                    observation=observation,
                    traj_dir=traj_dir,
                    return_type="info_trajectory",
                )

            save_predictions(traj_dir, instance_id, info)
            save_patch(traj_dir, instance_id, info)
            if args.actions.open_pr and should_open_pr(args, info, token=env._github_token):
                env.open_pr(trajectory=trajectory, push_gh_repo_url=args.actions.push_gh_repo_url)

        except KeyboardInterrupt:
            logger.info("Exiting InterCode environment...")
            env.close()
            break
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"❌ Failed on {env.record['instance_id']}: {e}")
            if args.raise_exceptions:
                raise e
            env.reset_container()
            continue
    
    # Close the environment
    if num_processes >1:
        env.close()
        logger.info("🏁 Finished task " + str(index))




def should_open_pr(args: ScriptArguments, info: Dict[str, Any], *, token: str="") -> bool:
    """Does opening a PR make sense?"""
    if not info.get("submission"):
        logger.info("Not opening PR because submission was made.")
        return False
    if info["exit_status"] != "submitted":
        logger.info("Not opening PR because exit status was %s and not submitted.", info["exit_status"])
        return False
    try:
        issue = get_gh_issue_data(args.environment.data_path, token=token)
    except InvalidGithubURL:
        logger.info("Currently only GitHub is supported to open PRs to. Skipping PR creation.")
        return False
    if issue.state != "open":
        logger.info(f"Issue is not open (state={issue.state}. Skipping PR creation.")
        return False
    if issue.assignee:
        logger.info("Issue is already assigned. Skipping PR creation. Be nice :)")
        return False
    if issue.locked:
        logger.info("Issue is locked. Skipping PR creation.")
        return False
    org, repo, issue_number = parse_gh_issue_url(args.environment.data_path)
    associated_commits = get_associated_commit_urls(org, repo, issue_number, token=token) 
    if associated_commits:
        commit_url_strs = ", ".join(associated_commits)
        if args.actions.skip_if_commits_reference_issue:
            logger.info(f"Issue already has associated commits (see {commit_url_strs}). Skipping PR creation.")
            return False
        else:
            logger.warning(
                "Proceeding with PR creation even though there are already commits "
                f"({commit_url_strs}) associated with the issue. Please only do this for your own repositories "
                "or after verifying that the existing commits do not fix the issue."
            )
    return True


def save_arguments(traj_dir: Path, args: ScriptArguments) -> None:
    """Save the arguments to a yaml file to the run's trajectory directory."""
    log_path = traj_dir / "args.yaml"

    if log_path.exists():
        try:
            other_args = args.load_yaml(log_path)
            if (args.dumps_yaml() != other_args.dumps_yaml()):  # check yaml equality instead of object equality
                logger.warning("**************************************************")
                logger.warning("Found existing args.yaml with different arguments!")
                logger.warning("**************************************************")
        except Exception as e:
            logger.warning(f"Failed to load existing args.yaml: {e}")

    with log_path.open("w") as f:
        args.dump_yaml(f)


def should_skip(args: ScriptArguments, traj_dir: Path, instance_id: str) -> bool:
    """Check if we should skip this instance based on the instance filter and skip_existing flag."""
    # Skip instances that don't match the instance filter
    if re.match(args.instance_filter, instance_id) is None:
        logger.info(f"Instance filter not matched. Skipping instance {instance_id}")
        return True

    # If flag is set to False, don't skip
    if not args.skip_existing:
        return False

    # Check if there's an existing trajectory for this instance
    log_path = traj_dir / (instance_id + ".traj")
    if log_path.exists():
        with log_path.open("r") as f:
            data = json.load(f)
        # If the trajectory has no exit status, it's incomplete and we will redo it
        exit_status = data["info"].get("exit_status", None)
        if exit_status == "early_exit" or exit_status is None:
            logger.info(f"Found existing trajectory with no exit status: {log_path}")
            logger.info("Removing incomplete trajectory...")
            os.remove(log_path)
        else:
            logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
            return True
    return False


def save_predictions(traj_dir: Path, instance_id: str, info, index=None):
    if index is not None:
        output_file = traj_dir / f"all_preds_{index}.jsonl"
    else:
        output_file = traj_dir / "all_preds.jsonl"
    model_patch = info["submission"] if "submission" in info else None
    datum = {
        KEY_MODEL: Path(traj_dir).name,
        KEY_INSTANCE_ID: instance_id,
        KEY_PREDICTION: model_patch,
    }
    with open(output_file, "a+") as fp:
        print(json.dumps(datum), file=fp, flush=True)
    logger.info(f"Saved predictions to {output_file}")


def save_patch(traj_dir: Path, instance_id: str, info) -> Optional[Path]:
    """Create patch files that can be applied with `git am`.
    
    Returns:
        The path to the patch file, if it was saved. Otherwise, returns None.
    """
    patch_output_dir = traj_dir / "patches"
    patch_output_dir.mkdir(exist_ok=True, parents=True)
    patch_output_file = patch_output_dir / f"{instance_id}.patch"
    if not "submission" in info:
        logger.info("No patch to save.")
        return
    model_patch = info["submission"]
    patch_output_file.write_text(model_patch)
    _print_patch_message(patch_output_file)
    return patch_output_file


def _print_patch_message(patch_output_file: Path):
    console = rich.console.Console()
    msg = [
        "SWE-agent has produced a patch that it believes will solve the issue you submitted!",
        "Use the code snippet below to inspect or apply it!"
    ]
    panel = rich.panel.Panel.fit(
        "\n".join(msg),
        title="🎉 Submission successful 🎉",
    )
    console.print(panel)
    content = [
        "```bash",
        f"# The patch has been saved to your local filesystem at:",
        f"PATCH_FILE_PATH='{patch_output_file.resolve()}'",
        "# Inspect it:",
        "cat \"${PATCH_FILE_PATH}\"",
        "# Apply it to a local repository:",
        f"cd <your local repo root>",
        "git apply \"${PATCH_FILE_PATH}\"",
        "```",
    ]
    console.print(rich.markdown.Markdown("\n".join(content)))


def get_args(args=None) -> ScriptArguments:
    """Parse command line arguments and return a ScriptArguments object.
    
    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    defaults = ScriptArguments(
        suffix="",
        num_tasks=-1,
        start_index=0, # NOTE: used for debug
        use_gold_patches=False, # NOTE: used for debug,
        num_processes=1,
        exp_subdir="hepllm-v0.3",
        environment=EnvironmentArguments(
            image_name="sweagent/swe-agent:latest",
            data_path="princeton-nlp/SWE-bench_Lite",
            split="dev",
            verbose=True,
            install_environment=True, # NOTE: the agent will be provided the default environment (see self.install_env in swe_env.py for more details)
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name="gpt4",
                total_cost_limit=0.0,
                per_instance_cost_limit=2.0,
                temperature=0.0,
                top_p=0.95,
            ),
            config_file="config/default.yaml",
            use_hepllm=False,
            hepllm_levels=1,
        ),
        actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
        filter_gold_patch_positives=False,
        gold_patch_results_file=GOLD_TEST,
    )

    # Nicer yaml dumping of multiline strings
    def multiline_representer(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, multiline_representer)

    return parse(ScriptArguments, default=defaults, add_config_path_arg=False, args=args)

def save_combined_predictions(args):
    """
    combines all the predictions files into one
    """
    all_preds_files = list(Path("trajectories") / Path(args.exp_subdir) / args.run_name).glob("all_preds_*.jsonl")
    all_preds_file = Path("trajectories") / Path(args.exp_subdir) / args.run_name / "all_preds.jsonl"
    # combine all the files into one
    with open(all_preds_file, "w") as outfile:
        for file in all_preds_files:
            with open(file, "r") as infile:
                for line in infile:
                    outfile.write(line)

if __name__ == "__main__":
    args = get_args()

    # check num processes <= 4
    assert args.num_processes <= 4, "num_processes must be less than or equal to 4"

    if args.num_processes == 1:
        main(args)
    else:
        with Pool(args.num_processes) as p:
            # note main (args, index, num_processes) is the function to be run in parallel
            p.starmap(main, [(args, i, args.num_processes) for i in range(args.num_processes)])

        # save the predictions in a joint all_preds file
        save_combined_predictions(args)
