class PromptConstructor:

    DEFAULT_TEMPLATE = {'system_prefix': '''SYSTEM: {}''',
                        'user_prefix': 'USER: {}',
                        'data_prefix': 'text: {}',
                        'response_prefix': 'ASSISTANT: '}

    def __init__(self, template=None):
        
        self.template = template if template else self.DEFAULT_TEMPLATE

    def _validate_inputs(self, examples=None):

        # validate examples
        if examples is not None:

            if not isinstance(examples, list):
                raise ValueError("Examples must be a list of dictionaries.")
            if len(examples) == 0:
                raise ValueError("Examples cannot be an empty list. Please do not pass a value for examples if you do not wish to use this option.")
            
            for example in examples:
                if "question" not in example.keys(): 
                    raise ValueError("One or more examples provided is missing a question. Please validate that each example dictionary contains the proper keys.")
                if "answer" not in example.keys(): 
                    raise ValueError("One or more examples provided is missing an answer. Please validate that each example dictionary contains the proper keys.")
                if not isinstance(example, dict):
                    raise ValueError("Examples must be a list of dictionaries.")

        # validate template
        if self.template:
            if not isinstance(self.template, dict):
                raise ValueError("Template must be a dictionary")
            if "system_prefix" not in self.template.keys():
                raise ValueError("Template dictionary is missing a system_prefix. The value for this key can be an empty string, but the key cannot be missing from the template. Here is an example of a well-formed template: {}".format(self.DEFAULT_TEMPLATE))
            if "user_prefix" not in self.template.keys():
                raise ValueError("Template dictionary is missing a user_prefix. The value for this key can be an empty string, but the key cannot be missing from the template. Here is an example of a well-formed template: {}".format(self.DEFAULT_TEMPLATE))
            if "data_prefix" not in self.template.keys():
                raise ValueError("Template dictionary is missing a data_prefix. The value for this key can be an empty string, but the key cannot be missing from the template. Here is an example of a well-formed template: {}".format(self.DEFAULT_TEMPLATE))
            if "response_prefix" not in self.template.keys():
                raise ValueError("Template dictionary is missing a response_prefix. The value for this key can be an empty string, but the key cannot be missing from the template. Here is an example of a well-formed template: {}".format(self.DEFAULT_TEMPLATE))

    def _combine_prompt_components(self, template, question, instructions=None, data=None, answer=None, prompt_component_separator='\n'):
        
        output = ''
        output += template['user_prefix'].format(question) + prompt_component_separator

        if instructions:
            output += instructions + prompt_component_separator 

        if data:
            output += template['data_prefix'].format(data)
            output += prompt_component_separator
        
        output += template['response_prefix']
        
        if answer:
            output += answer + prompt_component_separator
        
        return output

    def construct_prompt(self, question, instructions=None, data=None, examples=None, system_message=None, prompt_component_separator='\n'):

        self._validate_inputs(examples)
        
        prompt = ''
        
        if system_message:
            prompt += self.template['system_prefix'].format(system_message) + prompt_component_separator

        if examples:
            for example in examples:
                example_data=example["data"] if "data" in example.keys() else None
                prompt += self._combine_prompt_components(self.template, question=example["question"], instructions=instructions, 
                                                         data=example_data, answer=example["answer"], prompt_component_separator=prompt_component_separator)
        
        prompt += self._combine_prompt_components(self.template, question=question, instructions=instructions, data=data, answer=None, 
                                                        prompt_component_separator=prompt_component_separator)

        return prompt
