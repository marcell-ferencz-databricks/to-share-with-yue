# Databricks notebook source
# MAGIC %pip install --upgrade dbtunnel[gradio] pydantic langchain-openai langchain-community langchain mlflow faiss-cpu
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import gradio as gr
import os
import time
import re
import json
import mlflow

# COMMAND ----------

# env variables
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
os.environ["OPENAI_API_VERSION"] = "api-version (date)"
os.environ["AZURE_OPENAI_ENDPOINT"] = "your-azure-openai-endpoint"
os.environ["OPENAI_API_TYPE"] = "azure" # don't change this
os.environ["OPENAI_DEPLOYMENT_NAME"] = "the deployment name of the GPT4 endpoint"
os.environ["OPENAI_API_KEY"] = "same azure openai api key as the first line"

# COMMAND ----------

# change this to the run id of the best run in your experiment
BEST_RUN_ID = "run_id of the best run in your experiment" # change this
loaded_model = mlflow.langchain.load_model(f"runs:/{BEST_RUN_ID}/retrieval_qa")
loaded_model.invoke("What is the framework about really?") # test model

# COMMAND ----------

# MAGIC %md
# MAGIC # Create table to log to
# MAGIC
# MAGIC CHANGE THE SCHEMA NAME TO YOUR SCHEMA HERE

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS mde_ai.chat_feedback (
# MAGIC   id STRING,
# MAGIC   date DATE,
# MAGIC   user STRING,
# MAGIC   user_message STRING,
# MAGIC   bot_message STRING,
# MAGIC   liked BOOLEAN
# MAGIC )

# COMMAND ----------

import gradio as gr
from typing import List, Tuple



def vote(value, like_data: gr.LikeData):
    user_question = value[-1][0]
    user_question = user_question.replace('"', "'")

    bot_response = like_data.value
    bot_response = bot_response.replace('"', "'")

    # CHANGE THE SCHEMA NAME TO YOUR SCHEMA
    spark.sql(f"""
              INSERT INTO marcell.mde_ai.chat_feedback
              (id, date, user, user_message, bot_message, liked)
              (SELECT * FROM VALUES (sha2(concat(current_user(), current_timestamp()), 0), current_timestamp(), current_user(), "{user_question}", "{bot_response}", "{like_data.liked}"))
              """)

    return None

def format_sources(sources):


  sources_formatted = [("..." + d.page_content + "...", d.metadata["title"]) for d in sources]


  source_outputs = ""
  for i, source in enumerate(sources_formatted):
    split_source = source[0].split("\n")
    source_joined = "\n > ".join(split_source)
    source_output = f"""
  ### Source {i+1}:

  **Document: {source[1]}**

  > {source_joined}


  

  ================================================================================================================
    """
    source_outputs += source_output

  return source_outputs


def chat_response(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str]:
    # Call your model
    response = loaded_model.invoke(message)
    
    # Extract answer and sources
    answer = response["answer"]
    sources = response["context"]
    sources = format_sources(sources)
     
    # Update history with the new message pair
    history.append((message, answer))
    
    # Return the updated history and sources
    return history, sources

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Ask Questions about the Crown Estate's Health & Safety Policy")
    chatbot = gr.Chatbot(label="H&S Assistant", show_label=True)
    msg = gr.Textbox(label="Enter your question", placeholder="Do I have access to any funds for home office equipment?")
    clear = gr.Button("Clear")
    sources_box = gr.Markdown(label="Sources")
    chatbot.like(vote, chatbot)
    msg.submit(chat_response, 
               inputs=[msg, chatbot], 
               outputs=[chatbot, sources_box])
    clear.click(lambda: ([], ""), 
                outputs=[chatbot, sources_box])

# COMMAND ----------

from dbtunnel import dbtunnel
dbtunnel.gradio(demo).run()
