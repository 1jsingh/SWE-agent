# @yaml
# signature: |-
#   subtask_execute <agent_name>
#   <subtask_instruction>
#   end_subtask
# docstring: delegates and executes a subtask using another agent. For instance, when solving a github issue, the main agent can work in high level abstraction by delegating simpler tasks to subagents (e.g., reproduce github issue, find the root cause, etc.). The subtask instruction should include 1) subtask instruction (e.g., reproduce github issue) and 2) all information required by the subtask agent to execute the subtask. Remember the subtask agent starts with no memory and therefore has no access to what has been done before.
# end_name: end_subtask
# arguments:
#   agent_name:
#     type: string
#     description: a suitable name for the agent to execute the subtask
#     required: true
#   subtask_instruction:
#     type: string
#     description: should include 1) subtask instruction (e.g., reproduce github issue) and 2) all information required by the subtask agent to execute the subtask. Remember the subtask agent starts with no memory and therefore has no access to what has been done before.
#     required: true
subtask_execute() {
    echo "<<SUBTASK||"
    echo "AGENT: $1"
    echo "SUBTASK_INSTRUCTION: $2"
    echo "||SUBTASK>>"
}