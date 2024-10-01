# Databricks notebook source
# MAGIC %pip install --upgrade dbtunnel[gradio] gradio pydantic langchain-openai langchain-community langchain mlflow faiss-cpu textstat
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
# MAGIC # Define basic Gradio Chat Interface

# COMMAND ----------

with gr.Blocks() as demo:
    gr.Markdown("""

## Ask Questions about the Government's Generative AI framework whitepaper.

#### Questions are compared against paragraphs in the document to find which are the most relevant. The LLM then uses these pages as the basis for its answer.

""")
    
    chatbot=gr.Chatbot(height="70%")
    msg = gr.Textbox(label="Chat")
    clear = gr.ClearButton([msg, chatbot])

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def respond(chat_history):
        bot_message = loaded_model.invoke(chat_history[-1][0])
        chat_history[-1][1] = ""
        for character in bot_message:
            chat_history[-1][1] += character
            time.sleep(0.05)
            yield chat_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        respond, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    


    
demo.queue()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Gradio App on cluster
# MAGIC
# MAGIC Click the generated link

# COMMAND ----------

from dbtunnel import dbtunnel
dbtunnel.gradio(demo).run()
