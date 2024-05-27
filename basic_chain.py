import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

from dotenv import load_dotenv

MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ZEPHYR_ID = "HuggingFaceH4/zephyr-7b-beta"
# LLAMA_ID = "meta-llama/Meta-Llama-3-8B"
# ZEPHYR_ID_2 = "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"

def get_model(repo_id=ZEPHYR_ID, **kwargs):
    if repo_id == "ChatGPT":
        chat_model = ChatOpenAI(temperature=0, **kwargs)
    else:
        huggingfacehub_api_token = kwargs.get("HUGGINGFACEHUB_API_TOKEN", None)
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
        os.environ["HF_TOKEN"] = huggingfacehub_api_token

        llm = HuggingFaceHub(
            repo_id=repo_id,
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.1,
                "repetition_penalty": 1.03,
                "huggingfacehub_api_token": huggingfacehub_api_token,
            })
        chat_model = ChatHuggingFace(llm=llm)
    return chat_model