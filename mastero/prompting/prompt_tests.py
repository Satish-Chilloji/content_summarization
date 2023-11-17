import pytest
from prompt_constructor import PromptConstructor

well_formed_test_cases = [

# question only
{
"question": "What is data science?",
"expected_output": 
"""USER: What is data science?
ASSISTANT: """
},

# question only with system message
{
"question": "What is data science?",
"system_message": "You are a helpful, respectful and honest assistant.",
"expected_output": 
"""SYSTEM: You are a helpful, respectful and honest assistant.
USER: What is data science?
ASSISTANT: """
},

# question + data
{
"question": "What is data science?",
"system_message": "You are a helpful, respectful and honest assistant.",
"data": "Data science is magic",

"expected_output": 
"""SYSTEM: You are a helpful, respectful and honest assistant.
USER: What is data science?
text: Data science is magic
ASSISTANT: """
},

# data + single example
{
"question": "What is the color of the ball?",
"system_message": "You are a helpful, respectful and honest assistant.",
"data": "The ball is green",

"examples":         [
                    {'question': 'What is the color of the ball?',
                    'answer': 'red',
                    'data': 'The ball is red'
                    }
                    ],

"expected_output": 
"""SYSTEM: You are a helpful, respectful and honest assistant.
USER: What is the color of the ball?
text: The ball is red
ASSISTANT: red
USER: What is the color of the ball?
text: The ball is green
ASSISTANT: """
},

# data + multiple examples
{
"question": "What is the color of the ball?",
"system_message": "You are a helpful, respectful and honest assistant.",
"data": "The ball is green",

"examples":         [
                    {'question': 'What is the color of the ball?',
                    'answer': 'red',
                    'data': 'The ball is red'
                    },
                    {'question': 'What is the color of the ball?',
                    'answer': 'yellow',
                    'data': 'The ball is yellow'
                    }
                    ],

"expected_output": 
"""SYSTEM: You are a helpful, respectful and honest assistant.
USER: What is the color of the ball?
text: The ball is red
ASSISTANT: red
USER: What is the color of the ball?
text: The ball is yellow
ASSISTANT: yellow
USER: What is the color of the ball?
text: The ball is green
ASSISTANT: """
},

# instructions only
{
"question": "Tell me about basketball.",
"system_message": "You are a helpful, respectful and honest assistant.",
"instructions": "Reply conversationally and be as detailed as possible.",

"expected_output": 
"""SYSTEM: You are a helpful, respectful and honest assistant.
USER: Tell me about basketball.
Reply conversationally and be as detailed as possible.
ASSISTANT: """
},

# instructions and data
{
"question": "Tell me about basketball.",
"system_message": "You are a helpful, respectful and honest assistant.",
"data": "Basketball is a sport played across the world. It was invented in 1891 by James Naismith.",
"instructions": "Reply conversationally and be as detailed as possible.",

"expected_output": 
"""SYSTEM: You are a helpful, respectful and honest assistant.
USER: Tell me about basketball.
Reply conversationally and be as detailed as possible.
text: Basketball is a sport played across the world. It was invented in 1891 by James Naismith.
ASSISTANT: """
},

# template gets passed 
{"question": "What is data science?",
"template": {"system_prefix": "<sys> {} </sys>",
              "user_prefix": "{}",
              "data_prefix": "",
              "response_prefix": ""
              },
"system_message": "Respond like a data scientist",

"expected_output": 
"""<sys> Respond like a data scientist </sys>
What is data science?
"""
}
]

malformed_test_cases = [
{"data": "Data science is magic",
 "expected_error": TypeError}, # no question passed

{"question": "What is data science?",
"data": "Data science is magic",
 "template": {"system_prefix": "a", "user_prefix": "b", "data_prefix": "c"},
 "expected_error": ValueError}, # template missing response prefix

{"question": "What is data science?",
"data": "Data science is magic",
 "template": {"system_prefix": "a", "user_prefix": "b", "response_prefix": "c"},
 "expected_error": ValueError}, # template missing data prefix

{"question": "What is data science?",
"data": "Data science is magic",
 "template": {"system_prefix": "a", "data_prefix": "b", "response_prefix": "c"},
 "expected_error": ValueError}, # template missing user prefix

{"question": "What is data science?",
"data": "Data science is magic",
"template": {"data_prefix": "a", "user_prefix": "b", "response_prefix": "c"},
"expected_error": ValueError}, # template missing system prefix

{"question": "What is data science?",
"data": "Data science is magic",
"template": "data prefix",
"expected_error": ValueError}, # template is not a dictionary

{"question": "What is data science?",
"data": "Data science is magic",
"examples": [],
"expected_error": ValueError}, # examples is empty

{"question": "What is data science?",
"data": "Data science is magic",
"examples": "USER: Hello ASSISTANT: ",
"expected_error": ValueError}, # examples is not a list

{"question": "What is data science?",
"data": "Data science is magic",
"examples": [{"answer": "hello"}],
"expected_error": ValueError}, # examples is missing question

{"question": "What is data science?",
"data": "Data science is magic",
"examples": [{"question": "hello"}],
"expected_error": ValueError}, # examples is missing an answer
]


@pytest.mark.parametrize("test_case", well_formed_test_cases)
def test_well_formed_input(test_case):

    question=test_case["question"] 
    instructions=test_case["instructions"] if "instructions" in test_case.keys() else None
    data=test_case["data"] if "data" in test_case.keys() else None 
    examples=test_case["examples"] if "examples" in test_case.keys() else None 
    template=test_case["template"] if "template" in test_case.keys() else None 
    system_message=test_case["system_message"] if "system_message" in test_case.keys() else None
    prompt_component_separator=test_case["prompt_component_separator"] if "prompt_component_separator" in test_case.keys() else '\n'

    pc = PromptConstructor(template)  

    result = pc.construct_prompt(question, instructions, data, examples, system_message, prompt_component_separator)

    assert result == test_case["expected_output"]

@pytest.mark.parametrize("test_case", malformed_test_cases)
def test_malformed_input(test_case):

    print({key: value for key, value in test_case.items() if key != 'expected_error'})

    question=test_case["question"] if "question" in test_case.keys() else None
    instructions=test_case["instructions"] if "instructions" in test_case.keys() else None
    data=test_case["data"] if "data" in test_case.keys() else None 
    examples=test_case["examples"] if "examples" in test_case.keys() else None 
    template=test_case["template"] if "template" in test_case.keys() else None 
    system_message=test_case["system_message"] if "system_message" in test_case.keys() else None
    prompt_component_separator=test_case["prompt_component_separator"] if "prompt_component_separator" in test_case.keys() else '\n'

    with pytest.raises(Exception):
        
        pc = PromptConstructor(template=template)
        if question:  
            result = pc.construct_prompt(question, instructions, data, examples, system_message, prompt_component_separator)
        else:
            result = pc.construct_prompt(instructions=instructions, data=data, examples=examples, system_message=system_message, prompt_component_separator=prompt_component_separator)