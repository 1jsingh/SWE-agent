import os
import openai
from openai import OpenAI, AzureOpenAI
import re

###########################################################################################################
# Constants
############################################################################################################

# Azure API information
AZURE_API_INFO = {
# "https://gcrgpt4aoai9c.azurewebsites.net"
    "general-text": {
        "endpoint": "https://gcrgpt4aoai9c.azurewebsites.net",#"https://gcraoai9sw1.openai.azure.com/",#"https://gcrgpt4aoai9c.azurewebsites.net",
        "api_version": "2024-02-15-preview",
        "api_key": "oaip_YTSaVZxMelwBjdWGUSeltUOnCzuycXhY"#"oaip_PbqANzdXoagDWwgaiWyMNpaSJIlPTUng"#"oaip_kiCYfzMFFNSMwMPxCDXBOLMMmYcSHZYP" #oaip_YTSaVZxMelwBjdWGUSeltUOnCzuycXhY"
    },
    "general-text-v5": {
        "endpoint": "https://gcrendpoint.azurewebsites.net",
        "api_version": "2024-02-15-preview",
        "api_key": "oaip_bHmvbBqlIabwpckWsJnnsryCkqVafcoo"#"oaip_YTSaVZxMelwBjdWGUSeltUOnCzuycXhY"#"oaip_kiCYfzMFFNSMwMPxCDXBOLMMmYcSHZYP" #oaip_YTSaVZxMelwBjdWGUSeltUOnCzuycXhY"
    },
    
    "general-text-v2": {
        "endpoint": "https://gcrgpt4aoai5.openai.azure.com",
        "api_version": "2024-02-15-preview",
        "api_key": "653880d85b6e4a209206c263d7c3cc7a"
    },
    "general-text-v3": {
        "endpoint": "https://gcraoai5sw2.openai.azure.com",
        "api_version": "2024-02-15-preview",
        "api_key": "c4039886fa444b9fa0d945ff81e10f0a"
    },
    "general-text-v4": {
        "endpoint": "https://gcrgpt4aoai9c.azurewebsites.net/",
        "api_version": "2024-02-15-preview",
        "api_key": 'oaip_YTSaVZxMelwBjdWGUSeltUOnCzuycXhY'
    },
    "gpt4-turbo": {
        "endpoint": "https://test-gpt-4-turbo-australia-east.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "api_key": "b1485beab36d4796841878836f6b3575"
    }
}

###########################################################################################################
# Methods
############################################################################################################

def llm2api(llm_name, use_azure_api=True, verbose=True):
    """
    Converts the LLM name to the corresponding API model name.

    Args:
        llm_name (str): The name of the LLM.
        use_azure_api (bool, optional): Whether to use the Azure API model names. Defaults to True.
        verbose (bool, optional): Whether to print the model name. Defaults to True.

    Returns:
        str: The corresponding API model name.
    """
    if llm_name=='gpt4': 
        model_name = "gpt-4" if use_azure_api else "gpt-4-1106-preview"
    elif llm_name=='gpt4t': 
        model_name = 'gpt-4-turbo' if use_azure_api else "gpt-4-turbo-preview"
    elif llm_name=='gpt4-32k':
        model_name = 'gpt-4-32k' if use_azure_api else "gpt-4-turbo-preview"
    elif llm_name=='gpt3.5':
        model_name = "gpt-35-turbo" if use_azure_api else "gpt-3.5-turbo-0125"
    elif llm_name=='gpt3.5-16k':
        model_name = "gpt-35-turbo-16k" if use_azure_api else "gpt-3.5-turbo-0125"
    elif llm_name=='dpo': 
        model_name = "/root/models/dpo-rlaif-v0.1"
    elif llm_name=='mixtral': 
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    if verbose: 
        print(f"Using model: {model_name}")
    return model_name

###########################################################################################################
def create_agent(llm_name, openai_api_key, use_azure_api=True, azure_api_tag="gpt4-turbo", api_version="2024-02-15-preview"):
    """
    Creates an agent based on the specified language model name and API configuration.

    Args:
        llm_name (str): The name of the language model.
        openai_api_key (str): The API key for OpenAI.
        use_azure_api (bool, optional): Whether to use the Azure API. Defaults to True.
        azure_api_tag (str, optional): The tag for the Azure API. Defaults to "general-text".

    Returns:
        agent: An instance of the agent based on the specified language model and API configuration.
    """
    # Validate the Azure API tag
    valid_azure_api_tag_choices = AZURE_API_INFO.keys()
    if azure_api_tag not in valid_azure_api_tag_choices:
        raise ValueError(f"Invalid choice for azure_api_tag. Valid choices are {valid_azure_api_tag_choices}")

    if 'gpt' in llm_name:
        if use_azure_api:
            agent = AzureOpenAI(azure_endpoint=AZURE_API_INFO[azure_api_tag]["endpoint"],
                                api_version=api_version, #"2023-12-01-preview",
                                api_key=AZURE_API_INFO[azure_api_tag]["api_key"])
        else:
            agent = OpenAI(api_key=openai_api_key)
        

        # if use_azure_api and llm_name!='gpt4t' and False:
        #     agent = AzureOpenAI(azure_endpoint=AZURE_API_INFO[azure_api_tag]["endpoint"],
        #                     api_version=api_version, #"2023-12-01-preview",
        #                     api_key=AZURE_API_INFO[azure_api_tag]["api_key"])
        # elif use_azure_api and llm_name=='gpt4t' or True: # TODO: remove the "or True" condition for now using this default for gpt
        #     print ("using gpt4t")
        #     agent = AzureOpenAI(azure_endpoint=AZURE_API_INFO["gpt4-turbo"]["endpoint"],
        #                                         api_version=api_version,#"2023-07-01-preview",
        #                                         api_key=AZURE_API_INFO["gpt4-turbo"]["api_key"])
    elif 'dpo' in llm_name:
        agent = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
    elif 'mixtral' in llm_name:
        agent = OpenAI(base_url="http://localhost:8009/v1", api_key="-")
    return agent
