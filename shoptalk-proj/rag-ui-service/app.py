import gradio as gr
import pandas as pd
    
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
#from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

import openai
import os

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
def reverse_text(text):
    return text[::-1]

# Create the Gradio interface
iface = gr.Interface(fn=reverse_text, inputs="text", outputs="text")

# Launch the interface
iface.launch(server_name="0.0.0.0", server_port=7860)


