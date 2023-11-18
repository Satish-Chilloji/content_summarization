from llamaV2Ggml import llamaV2Ggml

if __name__=='__main__':    
    # Get model 
    model = llamaV2Ggml()
    model.download_model(local_download_path='./llama_2_models')