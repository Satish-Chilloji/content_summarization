import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class llamaV2Ggml:

    def __init__(self):
        
        self.llama_object = None

        self.DEFAULT_MODEL_LOAD_PARAMS = {
            'n_ctx': 2048,
            'n_threads': 2,  # CPU cores
            'n_batch': 512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            'n_gpu_layers': 32  # Change this value based on your model and your GPU VRAM pool.
        }
        self.DEFAULT_MODEL_GEN_INFERENCE = {
            'max_tokens': 1024,
            'temperature': 0.5,
            'top_p': 0.9,
            'repeat_penalty': 1.2,
            'top_k': 50,
            'echo': False
        }

        self.set_model_load_params = {}
        self.set_model_gen_params = {}

    def download_model(self, model_repo='TheBloke/Llama-2-7B-Chat-GGML',
                       model_choice='llama-2-7b-chat.ggmlv3.q6_K.bin', local_download_path='./llama_2_models'):
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.abspath(local_download_path), exist_ok=True)

        model_path = hf_hub_download(repo_id=model_repo, filename=model_choice,
                                     local_dir=local_download_path)

        return model_path

    def load_model(self, model_path, **kwargs):
        """
        model_path=model_path,
        n_ctx=2048,
        n_threads=8, # CPU cores
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=43
        Args:
            **kwargs:

        Returns:

        """
        self.set_model_load_params = self.DEFAULT_MODEL_LOAD_PARAMS.copy()
        self.set_model_load_params.update(kwargs)

        lcpp_llm = Llama(
            model_path=model_path,
            **self.set_model_load_params
        )
        self.llama_object = lcpp_llm

        return lcpp_llm

    def generate(self, prompt, **kwargs):
        
        if self.llama_object is None:
            raise IOError(f"Model object not found, load the model using `load_model()`first")

        self.set_model_gen_params = self.DEFAULT_MODEL_GEN_INFERENCE.copy()
        self.set_model_gen_params.update(kwargs)

        response = self.llama_object(
            prompt=prompt, **self.set_model_gen_params)

        print(response["choices"][0]["text"])

        return response
