import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import helper_functions as hf

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI()
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key='chat_history', k=5)

prompt = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template="""You are a kind agent, you help humans with real time questions 
    and you answer their questions with patience and politeness
    chat history: {chat_history}
    Human: {question}
    AI:""")

llmchain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

conversation_buffer = hf.ConversationSummaryBuffer()

# Streamlit App configurations
st.set_page_config(
    page_title='Dialog Agent UI',
    page_icon='..',
    layout='wide')

st.title('Dialog Agent UI')
st.write('I am your assistant. Ask me anything')

hf.init_db()

total_tokens = 0
amount_spent = 0

# Display last five conversations
for msg in conversation_buffer.get_last_five():
    if "Human" in msg:
        st.write(f"👤: {msg.split(': ')[1]}", unsafe_allow_html=True)
    else:
        st.write(f"🤖: {msg.split(': ')[1]}", unsafe_allow_html=True)

user_prompt = st.text_input("Enter your message here:")

if st.button('Submit'):
    conversation_buffer.add(f"Human: {user_prompt}")

    with get_openai_callback() as cb:
        ai_response = llmchain.predict(question=user_prompt)
        total_tokens = cb.total_tokens
        amount_spent = cb.total_cost

        conversation_buffer.add(f"AI: {ai_response}")
        hf.log_conversation(user_prompt, ai_response)

        user_prompt = ""  # Clear the input text after processing

st.sidebar.write(f"Total Tokens Used: {total_tokens}")
st.sidebar.write(f"Cost in USD: ${amount_spent}")

if st.sidebar.button('View Logs'):
    logs = hf.get_all_logs()
    for log in logs:
        st.write(f"👤: {log[1]}")
        st.write(f"🤖: {log[2]}")
