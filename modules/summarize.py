from modules.youtube_utils import get_transcript, process
from modules.llm_utils import setup_credentials, define_parameters, initialize_watsonx_llm
from modules.prompt_utils import create_summary_prompt
from langchain.chains import LLMChain

def summarize_video(video_url, language_code):
    if not video_url:
        return "Please provide a valid YouTube URL."
    
    fetched_transcript = get_transcript(video_url, language_code)
    if not fetched_transcript:
        return "No transcript available."
    
    processed_transcript = process(fetched_transcript)

    model_id, credentials, client, project_id = setup_credentials()
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
    prompt = create_summary_prompt()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({"transcript": processed_transcript})
