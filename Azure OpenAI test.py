# Databricks notebook source
# MAGIC %pip install --upgrade pydantic langchain-openai langchain-community langchain mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
import mlflow
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# COMMAND ----------

# env variables
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

# COMMAND ----------

os.environ["OPENAI_DEPLOYMENT_NAME"] = "your-deployment-name"

# COMMAND ----------

llm = AzureChatOpenAI(
    deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
    temperature=0.0,
    max_tokens=256
)
llm.invoke("Hi")

# COMMAND ----------

df_test_data = pd.read_csv("/Volumes/marcell/mde_ai/raw_files/ai_regulatory_test_data.csv")
df_test_data["answer"] = df_test_data["ground_truth"].copy() # mock perfect answer

# COMMAND ----------

os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_ENDPOINT

# COMMAND ----------

my_answer_similarity = mlflow.metrics.genai.answer_similarity(model="openai:/gpt-35-turbo") # correct model name

# COMMAND ----------

results = mlflow.evaluate(
    data=df_test_data,
    targets="ground_truth",
    predictions="answer",
    model_type="question-answering",
    evaluators="default",
    extra_metrics=[my_answer_similarity]
    )

# COMMAND ----------

results.metrics['answer_similarity/v1/mean']

# COMMAND ----------


