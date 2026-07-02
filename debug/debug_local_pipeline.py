from langchain_huggingface import HuggingFacePipeline
print('HuggingFacePipeline loaded')
try:
    pipe = HuggingFacePipeline.from_model_id(model_id='gpt2', task='text-generation', pipeline_kwargs={'max_new_tokens': 20})
    print('pipeline created', type(pipe))
    result = pipe.invoke('Hello world')
    print('result', result)
except Exception as e:
    import traceback
    print('error', type(e).__name__, e)
    traceback.print_exc()
