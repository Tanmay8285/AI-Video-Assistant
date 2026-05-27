from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistralai_api_key=os.getenv("MISTRALAI_API_KEY"),temperature=0.3)

def split_transcript(transcript:str)-> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return text_splitter.split_text(transcript)

def summarize(transcript:str)-> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant that summarizes transcripts of meetings."),
        ("human","Summarize the following transcript of a meeting:\n\n{transcript}")
    ])
    map_chain=prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"transcript":chunk}) for chunk in chunks]
    combined= "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant that summarizes transcripts of meetings."),  
        ("human","Summarize the following summaries of a meeting:\n\n{combined}")
    ])
    combined_chain=(
        RunnablePassthrough()|RunnableLambda(lambda x: {"combined":x})|combined_prompt|llm|StrOutputParser()
    )
    return combined_chain.invoke(combined)

def generate_title(transcript:str)-> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant that generates titles for meetings based on their transcripts."),
        ("human","Generate a concise and descriptive title for the following transcript of a meeting:\n\n{transcript}")
    ])
    chain=RunnablePassthrough()|RunnableLambda(lambda x: {"transcript":x})|prompt|llm|StrOutputParser()
    return chain.invoke(transcript[:2000])