from langchain.prompts import PromptTemplate

def create_summary_prompt():
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI asistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps.
    3. Focus on the spoken content ("Text").

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return PromptTemplate(input_variables=["transcript"], template=template)

def create_qa_prompt_template():
    template = """
    You are an expert assistant providing detailed answers based on the following video content.
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    Question: {question}
    """
    return PromptTemplate(input_variables=["context", "question"], template=template)
