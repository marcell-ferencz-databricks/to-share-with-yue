# Databricks notebook source
# MAGIC %pip install --upgrade pydantic langchain-openai langchain-community langchain mlflow textstat
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import langchain
import mlflow
from datetime import datetime
import os
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import openai

langchain.__version__, mlflow.__version__, openai.__version__

# COMMAND ----------

mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC # Set Schema and table names

# COMMAND ----------

SCHEMA = "default"
TABLE = "survey" #change this to the name of the table where you uploaded the survey CSV

# COMMAND ----------

SURVEY_DATA_TABLE_ADDRESS = f"{SCHEMA}.{TABLE}" #change this to the address of the table where you uploaded the survey CSV

# COMMAND ----------

# MAGIC %md
# MAGIC # Combine questions and responses into a single prompt per question
# MAGIC
# MAGIC This will create the following pattern for each question:
# MAGIC
# MAGIC > Question:
# MAGIC > `question`
# MAGIC > 
# MAGIC > Respondent 1:
# MAGIC > `response 1`
# MAGIC > 
# MAGIC > Respondent 2:
# MAGIC > `response 2`
# MAGIC > ...

# COMMAND ----------

df_survey = spark.table(SURVEY_DATA_TABLE_ADDRESS).toPandas()

def get_prompt(row):
  response_list = [f"Respondent {i+1}:\n" + row[f"respondent_{i+1}"] for i in range(len(row)-1)]

  response_formatted = "\n\n".join(response_list)

  prompt = f"Question: {row['question']}\n\nResponses:\n\n{response_formatted}"

  return prompt

df_survey["prompt"] = df_survey.apply(get_prompt, axis=1)

df_survey

# COMMAND ----------

# MAGIC %md
# MAGIC # Set Azure Open AI env vars

# COMMAND ----------

# env variables
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

# COMMAND ----------

# MAGIC %md
# MAGIC # Set parameters and prompt template

# COMMAND ----------

# parameters
CHAT_MODEL = "gpt-35-turbo-16k"
TEMPERATURE = 0.8
MAX_OUTPUT_TOKENS = 256
PROMPT_TEMPLATE = (
    "You are an assistant for text summarisation. "
    "Your task is to summarise a list of responses to a survey question"
    "from a list of humans. You will be given the question, and the" "individual responses, and you should generate a single short " "summary of all the responses."
    "Keep things to a few sentences, and don't include any context or " "details about the survey."
    "\n\n"
    "{prompt}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Instantiate llm model

# COMMAND ----------

llm = AzureChatOpenAI(
    deployment_name=CHAT_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_OUTPUT_TOKENS
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create prompt and chain

# COMMAND ----------

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain = prompt | llm | StrOutputParser()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run model on all questions and log to experiment

# COMMAND ----------

with mlflow.start_run():
  df_survey["summary"] = df_survey["prompt"].map(lambda x: chain.invoke({"prompt": x}))
  results = mlflow.evaluate(
        data=df_survey,
        predictions="summary",
        model_type="question-answering",
        evaluators="default"
        )
  
  mlflow.log_metrics(results.metrics)
  mlflow.log_params(
        {
            "prompt_template": PROMPT_TEMPLATE,
            "chat_model": CHAT_MODEL,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Save table

# COMMAND ----------

spark.createDataFrame(df_survey).drop("prompt").write.mode("overwrite").saveAsTable(f"{SCHEMA}.survey_summary")
