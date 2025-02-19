
import gradio as gr
import pandas as pd
    
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
#from langchain_community.llms import OpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

import openai
import os
import requests
import logging
from config import REVERSE_STRING_API_ENDPOINT, CREATE_INDEX_API_ENDPOINT, FAISS_VECTOR_DB_SEARCH_ENDPOINT
##############################
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Create the PromptTemplate
prompt_template=ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
    You are a virtual shopping assistant. Based on the customer’s query, recommend three relevant products from the provided list.
    Return a JSON object in the format:
    {{
      "products": [
        {{
          "product_name": "Product Name",
          "rephrased_description": "Engaging rephrased description.",
          "price": "Price",
          "image_id": "Image ID"
        }},
        ...
      ],
      "sales_pitch": "A short and engaging sales pitch to encourage purchase, personalized to the customer’s interests and customer's query."
    }}

    Product List: {product_list}
    Customer Query: "{customer_query}"
    """)
])

llm=ChatOpenAI(model="gpt-3.5-turbo")

# Create an LLMChain
shopping_chain=LLMChain(
    llm=llm,
    prompt=prompt_template
)
########
                  



# Configure logging to display messages on the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import requests
import logging


def fetch_top_items2(prompt):
    response = requests.post(FAISS_VECTOR_DB_SEARCH_ENDPOINT)
    if response.status_code == 200:
        print("FAISS index created successfully.")        
    else:
        print(f"Error creating FAISS index: {response.text}")        
    
    data = {"prompt": prompt}    
    print(f"Sending POST request to {FAISS_VECTOR_DB_SEARCH_ENDPOINT} with data: {data}")
    response = requests.post(FAISS_VECTOR_DB_SEARCH_ENDPOINT, json=data)
    print(f"response----------------: {response.text}")
    #response.raise_for_status()  # Raise an exception for HTTP errors
    try:
        response_data = response.json()  # Parse JSON response
    except requests.exceptions.JSONDecodeError:
        print("Error: Received invalid JSON response")
        response_data = None

    response_data = response.json()  # Parse JSON response
    print(f"Received response-------------: {response_data}")
    return response_data
    
 
def format_product_details(details):
    formatted_details = "\n".join([f"- **{key}**: {value}" for key, value in details.items()])
    return formatted_details   

    

def create_index():
    """
    Call the /create_index API to generate embeddings and store the FAISS index.
    """
    logging.debug("Triggering FAISS index creation.")
    response = requests.post(CREATE_INDEX_API_ENDPOINT)
    
    if response.status_code == 200:
        logging.info("FAISS index created successfully.")
        print("FAISS index created successfully.")
    else:
        logging.error(f"Error creating FAISS index: {response.text}")
        print(f"Error creating FAISS index: {response.text}")

def fetch_top_items(prompt):
    """
    Fetch top items from the backend API based on the user's shopping request.
    Returns a list of item IDs.
    """
    #api_endpoint = REVERSE_STRING_API_ENDPOINT
    api_endpoint = FAISS_VECTOR_DB_SEARCH_ENDPOINT
    data = {"prompt": prompt}

    logging.debug(f"Sending POST request to {api_endpoint} with data: {data}")
    response = requests.post(api_endpoint, json=data)
    print(f"response----------------: {response}")
    response.raise_for_status()  # Raise an exception for HTTP errors

    response_data = response.json()  # Parse JSON response
    logging.debug(f"Received response-------------: {response_data}")
    print(f"response_data--------------: {response_data}")

    # Extract the list of item IDs
    items_list = response_data.get("tmp_image_paths", [])

    # Log extracted items
    logging.debug(f"Extracted item IDs---------------: {items_list}")
    print(f"Extracted item IDs--------------: {items_list}")

    return items_list  # ✅ Return only the list of item IDs
      
   
       

def generate_llm_response(prompt, items):
    """
    Generate a response using the LLM that includes the list of items.
    """
    print(f"response----------- :{items}")
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
    #create_index()
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
        empty_df = pd.DataFrame(columns=["Top_K_IMAGE_PATHS"])
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
        fetch_top_items2,
        inputs=[query_input],
        outputs=[product_details_output, llm_response_output]
    )


################ SAAGAR ########
# Define the function for Gradio
def handle_query(query):
    # Dummy response for demonstration
    sales_pitch = f"Our top recommendations for: {query}!"
    print("==============================================================")
    #response_data = response.json()  # Parse JSON response
    api_response = fetch_top_items2(query)
    print("==============================================================")
    print(f'handle_query.api_response : {api_response}')
        
        
    base_image_path="/app/images/"
        
    product_data = [
        {
            "image": f"{base_image_path}{item['image_file_location']}",
            "details": {
                "Description": item['concatenated_desc'].split("nan")[0].strip(),  # Remove 'nan' from description
                "Price": f"${item['price']:.2f}",
                "Color": item['color_value'] if isinstance(item['color_value'], str)  else "Unknown" 
            }
        }
        for item in api_response
    ]   
        
    
    return sales_pitch, product_data


# Define the Gradio interface
# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("## ShopTalk - Your Personal Shopping Assistant")

    # Input for query
    query_input = gr.Textbox(label="Enter your shopping query")

    # Output for sales pitch
    sales_pitch_output = gr.Textbox(label="Sales Pitch", lines=4)

    # Horizontal row for product recommendations
    with gr.Row():
        product_outputs = []
        for i in range(3):  # Assuming 3 products
            with gr.Column():
                # Set fixed width and height for images
                product_image = gr.Image(label=f"Product {i+1} Image", height=450, width=450)  # Resize to 150x150
                product_details = gr.Markdown(label=f"Product {i+1} Details")
                product_outputs.append((product_image, product_details))

    # Button to trigger search
    submit_button = gr.Button("Search")

    # Define the function call for Gradio
    def update_ui(query):
        sales_pitch, product_data = handle_query(query)        
        outputs = [sales_pitch]

        # Update the product outputs
        for i in range(3):
            product = product_data[i]
            formatted_details = format_product_details(product["details"])
            outputs.extend([product["image"], formatted_details])

        return outputs

    # Connect input, outputs, and logic
    submit_button.click(
        update_ui,
        inputs=[query_input],
        outputs=[sales_pitch_output] + [output for pair in product_outputs for output in pair]
    )

# Launch the Gradio app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

