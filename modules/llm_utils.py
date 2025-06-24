from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, EmbeddingTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings

def setup_credentials():
    model_id = "meta-llama/llama-3-2-3b-instruct"
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key="qbHFz81kTQvaUjggp-8KdzlDPAMJfuTE6qnQMOcoRrcg"
    )
    client = APIClient(credentials)
    project_id = "09cc9801-1892-4aca-ac8f-fad0eb01ad8a"
    return model_id, credentials, client, project_id

def define_parameters():
    return {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 900,
    }

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.url,
        apikey=credentials.api_key,
        project_id=project_id,
        params=parameters
    )

def setup_embedding_model(credentials, project_id):
    return WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=credentials.url,
        apikey=credentials.api_key,
        project_id=project_id
    )
