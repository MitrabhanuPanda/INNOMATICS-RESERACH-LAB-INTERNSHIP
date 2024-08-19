import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import chromadb
import re
import os
from openai import OpenAI
import io



f = open("C:/Users/mitra/Desktop/INNOMATICS(MITRABHANU PANDA)/GEN AI (ADVANCE)/OPENAI KEY/openai_api_key.txt")
OPENAI_API_KEY = f.read()

client = OpenAI(api_key = OPENAI_API_KEY)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embeddeing(text):
    encoding = tokenizer.batch_encode_plus(
        [text],                    
        padding=True,           # Pad to the maximum sequence length   
        truncation=True,           # Truncate to the maximum sequence 
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state
    encoded_text = tokenizer.encode(text, return_tensors='pt')
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    tokenized_text = tokenizer.tokenize(decoded_text)
    sentence_embedding = word_embeddings.mean(dim=1)

    return sentence_embedding.tolist()[0]

def clean(text):
    text = text.lower()
    return re.sub("[^a-z A-Z]", "",text)
    
chroma_client = chromadb.PersistentClient(path="C:/Users/mitra/Desktop/INNOMATICS(MITRABHANU PANDA)/GEN AI (ADVANCE)/6. PROJECTS/5. CLONING SHAZAM/chroma_db")
collection = chroma_client.get_collection(name="Movies")


uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])
number = st.number_input('number of result',min_value=5,max_value=20)
search = st.button('Search')

if uploaded_file is not None:
    # Play the audio file
    audio=st.audio(uploaded_file, format="audio/mpeg", loop=True)
    


    # Convert the uploaded file to a BytesIO object
    audio_bytes = io.BytesIO(uploaded_file.getvalue())

    # Use the OpenAI API to transcribe the audio
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=uploaded_file, 
    response_format="text"
    )
    text=transcription




if search:
    query_emd = embeddeing(text)
    result = collection.query(
    query_embeddings=[query_emd],
    n_results=number,
            )
    for i in result['metadatas'][-1]:
                st.write(i['Movie ID'],i['name'])