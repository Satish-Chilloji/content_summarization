# Maestro

Your conductor for all things genAI. :magic_wand:

## Table of Contents
 
- [Usage & Installation](#usage--installation)
- [Components](#components)
- [Examples](#examples)


## Usage & Installation

This package should be run from within a **sagemaker notebook** with the following specifications:

- Image: PyTorch 2.0.0 Python 3.10 GPU Optimized
- Instance type: "ml.g5.2xlarge"

Before running your code please ensure the following commands are run (note that any shell commands run from the notebook need to prefaced with "!")

```
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install git+https://github.com/abetlen/llama-cpp-python.git@d644199fe8c014028c61738b6e7d5fe900fb4ba9 --force-reinstall --upgrade --no-cache-dir --verbose
```

In order to download this package in sagemaker you will need to first clone the magicshop repo in sagemaker.

Once you've completed these steps, you can add the following command to your notebook to install the package: ```!pip install -e <path to maestro folder>``` 

Note that the path will likely look something like: ../magicshop/projects/maestro

## Components

There are four main components to this library:

1. llm: This component contains code for downloading, loading, and calling llms locally. In the current version, only local llama2ggml is supported.
2. embeddings: This component contains code for downloading, loading, and calling embedding models. In the current version, only instruct embeddings are supported.
3. prompting: This component contains code for constructing prompts based on a template, using various components including questions, model instructions, data, few-shot examples, etc.
4. db_connection: This component contains code for interfacing with databases. In the current version, only s2 is supported.   

## Examples

Below are several examples of how to use the package.

**1. Downloading and loading the llama2 model.** This should only be done once otherwise you will likely run into gpu memory issues. The rest of the llm-related examples assume the llama model has been loaded.

```python
from maestro import llamaV2Ggml

model = llamaV2Ggml()
model.download_model(local_download_path='./llama_2_models')
model.load_model('llama_2_models/llama-2-13b-chat.ggmlv3.q6_K.bin')
``` 

**2. Making a llama2 call with a simple prompt**

```python
from maestro import PromptConstructor

# Construct prompt
query = "Can you teach me how Large Language Models work?"
prompt = PromptConstructor().construct_prompt(query)

# Call LLM
response = model.generate(prompt=prompt)

```

Output
```python
"""🤖 Sure, I'd be happy to help! Large language models (LLMs) are a type of artificial intelligence that can process and generate human-like text. They have become increasingly popular in recent years due to their impressive performance on a wide range of natural language processing tasks.

Here's an overview of how LLMs work:

1. Training Data: To train an LLM, you need a large dataset of text that the model can learn from. This dataset is typically composed of a variety of texts, such as books, articles, and websites, to ensure that the model is exposed to a wide range of writing styles and genres.
2. Model Architecture: The heart of an LLM is its architecture, which consists of multiple layers of interconnected nodes (also called "neurons"). Each node processes a small portion of the input text and passes it on to the next layer until the final output is generated.
3. Learning Process: During training, the model learns to predict the next word in a sequence of text based on the context provided by the previous words. This process is repeated millions of times, with the model adjusting its weights and biases to minimize the error between its predictions and the actual next word.
4. Generative Process: Once trained, an LLM can be used to generate new text that is similar in style and structure to the training data. The generative process involves feeding a prompt into the model's input layer and allowing it to generate text one word at a time until the desired length is reached.
5. Attention Mechanism: One key component of LLMs is their attention mechanism, which allows the model to focus on specific parts of the input text when generating output. This helps ensure that the generated text is relevant and coherent.
6. Fine-tuning: To improve the performance of an LLM on a specific task (such as language translation or sentiment analysis), it can be fine-tuned by adjusting its weights and biases to better fit the new task. This process typically involves retraining the model on a smaller dataset that is tailored to the new task.

I hope this helps give you an understanding of how LLMs work! Do you have any specific questions or areas you'd like me to expand upon? 🤔
"""
```

**3. Making a llama2 call with a more complex few-show prompt.** Note that in practice your data would likely be generated from some other process (for example a call to a database).

```python
from maestro import PromptConstructor

# Construct prompt
query = "Can you please provide a summary of the following data?"

# https://towardsdatascience.com/transformers-141e32e69591
data = """Transformers are a type of neural network architecture that have been gaining popularity. Transformers were recently used by OpenAI in their language models, and also used recently by DeepMind for AlphaStar — their program to defeat a top professional Starcraft player.

Transformers were developed to solve the problem of sequence transduction, or neural machine translation. That means any task that transforms an input sequence to an output sequence. This includes speech recognition, text-to-speech transformation, etc.."""

examples = [{"question": "Can you please provide a summary of the following data?",
            
            # https://www.techtarget.com/whatis/definition/large-language-model-LLM
           "data": """A large language model (LLM) is a type of artificial intelligence (AI) algorithm that uses deep learning techniques and massively large data sets to understand, summarize, generate and predict new content. The term generative AI also is closely connected with LLMs, which are, in fact, a type of generative AI that has been specifically architected to help generate text-based content.

            Over millennia, humans developed spoken languages to communicate. Language is at the core of all forms of human and technological communications; it provides the words, semantics and grammar needed to convey ideas and concepts. In the AI world, a language model serves a similar purpose, providing a basis to communicate and generate new concepts.""",
            
            "answer": "LLMs use deep learning to generate new content and provide a basis to communicate much like humnas."

            }]

instructions = "Please limit your response to one sentence."

pc = PromptConstructor()

prompt = pc.construct_prompt(question=query, instructions=instructions, data=data, examples=examples)

# Call LLM
response = model.generate(prompt=prompt) 
```

Output
```python
"""Transformers are a type of neural network architecture used for tasks like language modeling and speech recognition.
"""
```

4. **Create a singlestore database table**

```python
from maestro.db_connection import SingleStoreConnector

# replace with credentials
s2_host = os.environ['S2_STAGING_HOST']
s2_port = os.environ['S2_STAGING_PORT']
s2_user = os.environ['S2_USER']
s2_pwd = os.environ['S2_PASSWORD']
db = 'data_science'

# get cursor
s2_con = SingleStoreConnector.get_connection(host=s2_host, port=s2_port, database=db, user=s2_user, password=s2_pwd)
s2_cur = s2_con.cursor()

# drop and create queries
drop_table = ''' drop table if exists testing_table '''
create_table = '''
    create table if not exists testing_table
        (
        sentence Varchar(256),
        embedding blob
        )
    '''

# execute queries
s2_cur.execute(drop_table)
s2_cur.execute(create_table)
```

5. **Embed vectors and upload to singlestore (assumes singlestore table already exists)**

```python
from maestro import InstructEmbedder
import json
sentence1 = "I work in data"
sentence2 = "I am an investment banker"

# download embedding model
embedder = InstructEmbedder(download_model=True)

# get embeddings for sentences 
embedding_instruction = 'Represent the statement for entity classification:'
embedding1 = embedder.get_embeddings(str_or_str_list=sentence1, instruction=embedding_instruction)[0]
embedding2 = embedder.get_embeddings(str_or_str_list=sentence2, instruction=embedding_instruction)[0]

# convert embedding to json
embedding_as_json1 = json.dumps(embedding1.tolist())
embedding_as_json2 = json.dumps(embedding2.tolist())

# insert sentences and embeddings to singlestore table (in practice, you'll probably use a loop)
write_stmt = """
INSERT into testing_table(sentence, embedding)
VALUES(%(sentence)s, json_array_pack(%(embedding)s))
"""

s2_cur.execute(write_stmt, dict(sentence=sentence1, embedding=embedding_as_json1))
s2_cur.execute(write_stmt, dict(sentence=sentence2, embedding=embedding_as_json2))
```

6. **Get vector similarity between new embedding and embeddings stored in database (assumes example 4-5 have run)**

```python
import pandas as pd
new_sentence = "I am a data scientist"

# get the embeddings 
new_sentence_embedding = embedder.get_embeddings(str_or_str_list=new_sentence, instruction=embedding_instruction)[0]
new_sentence_embedding_as_json = json.dumps(new_sentence_embedding.tolist())

# perform similarity search vs. embedding column in database
similarity_search = f'''
    SELECT DOT_PRODUCT(JSON_ARRAY_PACK('{new_sentence_embedding_as_json}'), embedding) as cosine_sim, * 
    FROM testing_table
    ORDER BY 1 DESC
    '''

# execute query
s2_cur.execute(similarity_search)

# Fetch and show results of query
df = pd.DataFrame(s2_cur.fetchall(), columns = ['match', 'sentence', 'embedding'])
df[['match', 'sentence']]


```

Output

|    score | sentence                  |
|---------:|:--------------------------|
| 0.930934 | I work in data            |
| 0.843047 | I am an investment banker |

Our vector similarity search here does a nice job showing that "I am a data scientist" is closer to "I work in data" than "I am an investment banker".  