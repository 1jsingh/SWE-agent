# @yaml
# signature: |-
#   finish_subtask_and_report_to_main_agent
#   <subtask_completion_report>
#   end_report
# docstring: reports the completion of a subtask to the main agent. The subtask completion report should include all information required by the main agent to understand the completion of the subtask. Remember that the main agent has no memory of what has been done by the subtask agent.
# end_name: end_report
# arguments:
#   subtask_completion_report:
#     type: string
#     description: should include all information required by the main agent to understand the completion of the subtask
#     required: true
finish_subtask_and_report_to_main_agent() {
    echo "<<SUBTASK_REPORT||"
    echo "SUBTASK_COMPLETION_REPORT: $1"
    echo "||SUBTASK_REPORT>>"
}
