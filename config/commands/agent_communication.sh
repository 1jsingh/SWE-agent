# @yaml
# signature: |-
#   ask_question_to_subagent <agent_name>
#   <question>
#   end_question
# docstring: asks a question to another agent. The agent will respond with an answer. The question should be clear and concise.
# end_name: end_question
# arguments:
#   agent_name:
#     type: string
#     description: EXACT name of your prior subagent to whom you need to ask a question. This should be one of the sub_agents that you previously delegated a task to. 
#     required: true
#   question:
#     type: string
#     description: the question you want to ask the subagent. The question should be clear and concise.
#     required: true
ask_question_to_subagent() {
    echo "<<ASK_QUESTION_TO_SUBAGENT||"
    echo "AGENT: $1"
    echo "QUESTION: $2"
    echo "||ASK_QUESTION_TO_SUBAGENT>>"
}


# @yaml
# signature: |-
#   ask_question_to_parent_agent
#   <question>
#   end_question
# docstring: asks a question to the parent agent who gave you the subtask. The agent will respond with an answer. The question should be clear and concise.
# end_name: end_question
# arguments:
#   question:
#     type: string
#     description: the question you want to ask the parent agent. The question should be clear and concise.
#     required: true
ask_question_to_parent_agent() {
    echo "<<ASK_QUESTION_TO_PARENT_AGENT||"
    echo "AGENT: $1"
    echo "QUESTION: $2"
    echo "||ASK_QUESTION_TO_PARENT_AGENT>>"
}
