import streamlit as st
from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer  


model_check_point = 'gpt2'

llm = TFAutoModelForCausalLM.from_pretrained(model_check_point)
tokenizer = AutoTokenizer.from_pretrained(model_check_point) 
pipe = pipeline(tokenizer=tokenizer,model=llm) 




st.title('Chat Bot Service') 
input_text = st.text_input('Search the topic u want')    

if input_text:
    input_prompt = f'Question: {input_text}'
    response = pipe(input_prompt, max_length=512, num_return_sequences=1)
    st.write(response[0].strip()) 
