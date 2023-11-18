import re
import time
from prompt_constructor import PromptConstructor
from llamaV2Ggml import llamaV2Ggml

def main(input):
    # Construct prompt
    query = "Can you please provide a summary of the following data?"
    # https://towardsdatascience.com/transformers-141e32e69591
    data = f"""{input}"""

    examples = [{"question": "Can you please provide a summary of the following data?",

                # https://www.techtarget.com/whatis/definition/large-language-model-LLM
            "data": """A large language model (LLM) is a type of artificial intelligence (AI) algorithm that uses deep learning techniques and massively large data sets to understand, summarize, generate and predict new content. The term generative AI also is closely connected with LLMs, which are, in fact, a type of generative AI that has been specifically architected to help generate text-based content.
                Over millennia, humans developed spoken languages to communicate. Language is at the core of all forms of human and technological communications; it provides the words, semantics and grammar needed to convey ideas and concepts. In the AI world, a language model serves a similar purpose, providing a basis to communicate and generate new concepts.""",
                
                "answer": "LLMs use deep learning to generate new content and provide a basis to communicate much like humnas."
                }]
    instructions = "Please limit your response to two sentence."
    pc = PromptConstructor()
    prompt = pc.construct_prompt(question=query, instructions=instructions, data=data, examples=examples)

    model = llamaV2Ggml()
    model.load_model('llama_2_models/llama-2-7b-chat.ggmlv3.q5_1.bin')
    response = model.generate(prompt=prompt)
    return response

if __name__=='__main__':
    input="""Three senior OpenAI researchers Jakub Pachocki, Aleksander Madry and Szymon Sidor resigned. Three senior OpenAI researchers Jakub Pachocki, Aleksander Madry and Szymon Sidor told associates they have resigned, news agency Reuters reported. 
    The board of the company behind ChatGPT on Friday fired OpenAI CEO Sam Altman - to many, the human face of generative AI - sending shock waves across the tech industry.
    OpenAI's chief technology officer Mira Murati will serve as interim CEO, the company said, adding that it will conduct a formal search for a permanent CEO.
    The announcement also said another OpenAI co-founder and top executive, Greg Brockman, the board’s chairman, would step down from that role but remain at the company, where he serves as president.
    But later on X, formerly Twitter, Brockman posted a message he sent to OpenAI employees in which he wrote, “based on today’s news, i quit.”"""
    main()


