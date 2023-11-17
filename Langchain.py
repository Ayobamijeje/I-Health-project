import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from base import classifier
from keys import mykey
from tensorflow.keras.models import load_model
from langchain.llms import OpenAI
import os


os.environ['OPENAI_API_KEY'] = mykey

header = st.container()
history = st.container()
vitals = st.container()
exam = st.container()
test = st.container()
final = st.container()
imgbody = st.container()
Update = st.container()

with header:
	st.header('Welcome to I-Health :leo:')
	st.write('##### This page is about machine learning app involving health care dataset')

with history:
	History  = st.text_input(label = "How can i help you", placeholder="How can i help you")

with exam: 
	Examination = st.text_input(label = "Examination", placeholder="Examination")

with vitals: 
	Vitals = st.text_input(label = "Observation", placeholder="Observation")


with test: 
	Test = st.text_input(label = "Test result", placeholder="Test result")

with imgbody:
	file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])

	if file is not None:
		image = Image.open(file).convert('RGB')
		st.image(image)

		model = load_model('C:/Users/Home/OneDrive/Desktop/savedmode/chestxray.h5')



		predict = classifier(image, model)

		name = ' is Normal' if predict < 0.5 else 'Showed Pneumonia'
		named = 'Chest xray ' + name
		st.write(predict)

	else : 
		named = ''

			


with Update: 
	Update = st.text_input(label = "Update", placeholder="Update")
	
input_info = str(History + ' ' + Examination + ' ' + Vitals + ' ' + Test + ' ' + named + ' ' + Update)
print(type(input_info))



llm = OpenAI(temperature=0)


from langchain.prompts import PromptTemplate


language_prompt = PromptTemplate(
    input_variables= ['input_info'],
    template=" I am a doctor in accident and emergency and a patient comes with {input_info}"
)
#language_prompt.format(History=History,Examination=Examination)

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(input_key = 'input_info')

chain2=LLMChain(llm=llm,prompt=language_prompt, memory = memory)

me = chain2(input_info)
st.write(me)






		



