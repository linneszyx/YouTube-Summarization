from modules.youtube_utils import get_transcript, process, chunk_transcript
from modules.llm_utils import setup_credentials, define_parameters, initialize_watsonx_llm, setup_embedding_model
from modules.faiss_utils import create_faiss_index, retrieve
from modules.prompt_utils import create_qa_prompt_template
from langchain.chains import LLMChain

def generate_answer(video_url, user_question, language_code):
    if not video_url or not user_question:
        return "Please provide a valid video URL and question."

    fetched_transcript = get_transcript(video_url, language_code)
    if not fetched_transcript:
        return "No transcript available."
    
    processed_transcript = process(fetched_transcript)
    chunks = chunk_transcript(processed_transcript)

    model_id, credentials, client, project_id = setup_credentials()
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
    embedding_model = setup_embedding_model(credentials, project_id)
    faiss_index = create_faiss_index(chunks, embedding_model)

    qa_prompt = create_qa_prompt_template()
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt, verbose=True)

    relevant_context = retrieve(user_question, faiss_index, k=7)
    answer = qa_chain.predict(context=relevant_context, question=user_question)

    return answer
