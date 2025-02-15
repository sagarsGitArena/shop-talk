
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
import logging
from config import REVERSE_STRING_API_ENDPOINT
openai.api_key = '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'  # Replace with your actual OpenAI API key


# Configure logging to display messages on the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_top_items(prompt):
    """
    Fetch top items from the backend API based on the user's shopping request.
    """
    api_endpoint = REVERSE_STRING_API_ENDPOINT
    data = {"prompt": prompt}
    logging.debug(f"Sending POST request to {api_endpoint} with data: {data}")
    response = requests.post(api_endpoint, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    logging.debug(f"Received response: {response.text}")

    response_data = response.json()  # Correct way to parse JSON
    logging.debug(f"Parsed JSON response: {response_data}")
    print(f"response :{response_data}")
    print("Available keys in response:", response_data.keys())

    # Convert to DataFrame
    # Identify the correct key containing the records
    items_key = None
    for key in response_data.keys():
        if isinstance(response_data[key], list):  # The key holding the list of records
            items_key = key
            break

    if not items_key:
        print("No list of items found in response.")
        return pd.DataFrame(columns=["main_image_id"])

    # Extract the main_image_id from each record
    items = response_data[items_key]
    image_ids = [item.get("main_image_id", None) for item in items if "main_image_id" in item]

    items = response_data[items_key]
    image_ids = [item.get("main_image_id", None) for item in items if "main_image_id" in item]
    items_df = pd.DataFrame(image_ids, columns=["main_image_id"])
    print(f"items_df key identified :{items_df}")
    return items_df   
       

def generate_llm_response(prompt, items):
    """
    Generate a response using the LLM that includes the list of items.
    """
    item_list = "\n".join([f"- {item}" for item in items])
    llm_prompt = (
        f"User is looking for: {prompt}\n\n"
        f"Top recommended items:\n{item_list}\n\n"
        "Please generate a friendly response that includes these items."
    )
    response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": llm_prompt}
            ],
            max_tokens=150
        )
    llm_response = response.choices[0].message.content.strip()
    return llm_response       
    

def shopping_agent(prompt):
    items_list = fetch_top_items(prompt)  # Expecting a list of strings
    print(f"ids from shopping_agent :{items_list}")
     # Ensure `items_list` is a list, not a DataFrame
    if isinstance(items_list, pd.DataFrame):  # If mistakenly a DataFrame, extract values
        items_list = items_list["main_image_id"].tolist()

    if items_list:  # ✅ Now correctly checking for an empty list
        # Convert list into a DataFrame
        items_df = pd.DataFrame(items_list, columns=["main_image_id"])

        # Generate a response using the LLM
        llm_response = generate_llm_response(prompt, items_df)

        return items_df, llm_response
    else:
        # Return an empty DataFrame with only "main_image_id" column
        empty_df = pd.DataFrame(columns=["main_image_id"])
        return empty_df, "No items found for your request."


# Gradio Interface
with gr.Blocks(css="styles.css") as app:
    gr.Markdown("## ShopTalk - Your Personal Shopping Assistant")
    with gr.Row():
        query_input = gr.Textbox(label="Enter your shopping query")
    with gr.Row():
        product_details_output = gr.DataFrame(label="Product Details")
    with gr.Row():
        llm_response_output = gr.Textbox(label="LLM Response", interactive=False)    
    submit_button = gr.Button("Search")
    submit_button.click(
        shopping_agent,
        inputs=[query_input],
        outputs=[product_details_output, llm_response_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

