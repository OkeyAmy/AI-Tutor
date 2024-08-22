import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
os.environ['google_api_key'] = os.getenv('GOOGLEAI_API_KEY')

# Load the Google API key from environment variables
google_api_key = os.getenv('GOOGLEAI_API_KEY')

if google_api_key is None:
    st.error('Google AI API key is not set.')
    st.stop()

from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

st.title('AI Tutor')
st.write('Enter a topic that you will want to learn')  # Display the instruction
prompt  = st.text_input('Enter a topic')

# Prompt Templates
title_template =  PromptTemplate(
    input_variables= ['topic', 'content_description'],
    template= 'Write a title heading for this: {topic}. Summarise the article: {content_description}'
)

content_template =  PromptTemplate(
    input_variables= ['title', 'wikipedia_research'],
    template= """
Write an extensive article starting with a sub-title relevant to the title and an introduction following it.: {title}, including examples and code samples, if necessary, while leveraging Wikipedia for research {wikipedia_research}.
"""
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
content_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
llm = GoogleGenerativeAI(temperature=0.6, model='gemini-1.5-flash', google_api_key=google_api_key)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', 
                       memory=title_memory)
content_chain = LLMChain(llm=llm, prompt=content_template, verbose=True,output_key='content', 
                        memory=content_memory)

# Wikipedia API
wiki = WikipediaAPIWrapper()

wiki_research = None  # Initialize outside the conditional block
content_description = None
title = None
script = None

#Show stuffs to the screen if theres a prompt
if st.button('Ask'):
    if prompt:
        wiki_research = wiki.run(prompt)
        script = content_chain.run({"title":prompt, "wikipedia_research":wiki_research})
        content_description = llm(f"Summarize this article in a few sentences: {script}")
        title = title_chain.run({'topic':prompt, 'content_description':content_description})
        st.write(title)
        st.write(script)
    else:
        st.write('Enter a topic that you want to learn')

st.sidebar.header('Knowledge Base')  # Move header outside the 'with' block
with st.sidebar:
    # The expander now always shows 
    with st.expander('Title History'):
        if title_memory.buffer:
            st.info(title_memory.buffer)
    with st.expander('Content History'):
        if content_memory.buffer:
            st.info(content_memory.buffer)
    with st.expander('Wikipedia Research History'):
        if wiki_research:  # Only display if there's research
            st.info(wiki_research)
