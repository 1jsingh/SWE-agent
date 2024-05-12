# @yaml
# signature: |-
#   ask_question_to_parent_agent <parent_agent>
#   <question>
#   end_question
# docstring: asks a question to the parent agent who gave you the subtask. The agent will respond with an answer. The question should be clear and concise.
# end_name: end_question
# arguments:
#   parent_agent:
#     type: string
#     description: must be equal to 'parent_agent'. This is fixed and should not be changed.
#     required: true
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
