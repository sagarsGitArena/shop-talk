import gradio as gr
import pandas as pd
    
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
#from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

import openai
import os
import requests

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("## ShopTalk - Your Personal Shopping Assistant")
    with gr.Row():
        query_input = gr.Textbox(label="Enter your shopping query")
    with gr.Row():
        sales_pitch_output = gr.Textbox(label="Sales Pitch", lines=4)
    with gr.Row():
        product_images_output = gr.Gallery(label="Recommended Products")
    with gr.Row():
        product_details_output = gr.JSON(label="Product Details")

    submit_button = gr.Button("Search")
    submit_button.click(
       # handle_query,
        inputs=[query_input],
        outputs=[sales_pitch_output, product_images_output, product_details_output]
    )




# Define the function that will process the input
# def reverse_text(text):
#     return text[::-1]

#VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "http://vector-db-service:8000/string-reverse")
def reverse_text(input_text):       
    #api_url = VECTOR_DB_HOST
    api_url='http://vector-db-service:8000/string-reverse'
    print(f'api_url: {api_url}')
    payload = {"text": input_text}

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an error for HTTP failures
        result = response.json()  # Extract JSON response
        return result['reversed']
    except requests.exceptions.RequestException as e:
        return f"Error contacting API: {str(e)}"




# Create the Gradio interface
iface = gr.Interface(
    fn=reverse_text, 
    inputs="text", 
    outputs="text",
    title="String Reverser UI",
    description="Enter a string to reverse it using vector-db-service API."
)

# Launch the interface
iface.launch(server_name="0.0.0.0", server_port=7860)


