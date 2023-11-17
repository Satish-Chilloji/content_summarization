from InstructorEmbedding import INSTRUCTOR
from huggingface_hub import snapshot_download
import os

class InstructEmbedder:
    def __init__(self, model_path=None, download_model=False):
        """

        Args:
            model_path: The pre-existing model path.
            download_model: (Bool): If set to True the model will be downloaded in `self.default_download_path` variable path.
        """
        self.default_download_path = './instructor-large'

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.abspath(self.default_download_path), exist_ok=True)

        self.model_path = model_path

        if model_path:
            self.model_path = model_path

        elif download_model:
            model_path = snapshot_download(repo_id='hkunlp/instructor-large',
                                           local_dir=self.default_download_path)

            print(f"Model downloaded in {model_path}")

            self.model_path = model_path

        else:
            raise Exception("Model not found, either set download model or provide a model path")

        self.model = self._load_model(model_path=self.model_path)
        self.DEFAULT_INSTRUCTION = "Represent the statement for entity classification:"

        pass

    def _load_model(self, model_path):
        """
        Function to load the Instruct Embedding Model
        Args:
            model_path: Model Download path

        Returns:

        """
        return INSTRUCTOR(model_path)

    def _combine_instruction_strlist(self, instruction, str_list):
        """
        Helper function to combine instruction and vector
        Args:
            instruction:
            str_list:

        Returns:

        """
        return [[instruction, item] for item in str_list]

    def get_embeddings(self, str_or_str_list: list, instruction: str = ""):
        """
        Function to get instruct embeddings
        Args:
            str_or_str_list: Pass either
            instruction: Instruction for the embedding model.

        Returns:

        """
        if not instruction:
            instruction = self.DEFAULT_INSTRUCTION

        if isinstance(str_or_str_list, str):
            str_or_str_list = [str_or_str_list]

        encoding_data = self._combine_instruction_strlist(instruction, str_or_str_list)
        return self.model.encode(encoding_data)


if __name__ == '__main__':
    ie = InstructEmbedder(download_model=True)
    embeddings = ie.get_embeddings(str_or_str_list=["United States", "India"])
    embeddings = ie.get_embeddings(str_or_str_list='Apple Iphone')
