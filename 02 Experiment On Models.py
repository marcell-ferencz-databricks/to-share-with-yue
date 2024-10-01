# Databricks notebook source
# MAGIC %md
# MAGIC # Install Required Packages

# COMMAND ----------

# MAGIC %pip install --upgrade pydantic langchain-openai langchain-community langchain mlflow faiss-cpu textstat
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Packages

# COMMAND ----------

import langchain
import mlflow
from datetime import datetime
import os
import pandas as pd
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.document_loaders import PySparkDataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import openai

langchain.__version__, mlflow.__version__, openai.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC # Set Environment Variables
# MAGIC
# MAGIC These help communicate with Azure Open AI. Note we are also setting OpenAI env vars, but these are just to trick Mlflow into using the Azure Open AI endpoints as an LLM judge, instead of the Open AI defaults.

# COMMAND ----------

# env variables
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
os.environ["OPENAI_API_VERSION"] = "api-version (date)"
os.environ["AZURE_OPENAI_ENDPOINT"] = "your-azure-openai-endpoint"
os.environ["OPENAI_API_TYPE"] = "azure" # don't change this
os.environ["OPENAI_DEPLOYMENT_NAME"] = "the deployment name of the GPT4 endpoint"
os.environ["OPENAI_API_KEY"] = "same azure openai api key as the first line"

# COMMAND ----------

# MAGIC %md
# MAGIC # Enter Schema and raw filepaths

# COMMAND ----------

# change these...
SCHEMA = "your-schema" # Schema where your table with the parsed document(s) lives
RAW_FILE_PATH = "dbfs:/path/to/your/raw/csv" # Path to test CSV directory
vs_persist_directory = "langchain/faiss_index" # Path where the FAISS database will be saved -- don't need to change this one

# COMMAND ----------

# MAGIC %md
# MAGIC # Set experiment parameters
# MAGIC
# MAGIC This is what you'll mostly change for the retrieval:
# MAGIC * `CHUNK_SIZE`: how big a document chunk should be
# MAGIC * `CHUNK_OVERLAP`: how much overlap should be between adjacent chunks
# MAGIC * `EMBEDDING_MODEL`: the deployment name of the Azure OpenAI embedding model
# MAGIC * `NUM_CHUNKS`: how many chunks the retriever should retrieve based on the user query
# MAGIC
# MAGIC and for the LLM:
# MAGIC * `CHAT_MODEL`: the deployment name of the Azure OpenAI chat model to use
# MAGIC * `TEMPERATURE`: the temperature (i.e. 'randomness') of the response
# MAGIC * `MAX_OUTPUT_TOKENS`: maximum length of the response
# MAGIC * `SYSTEM_PROMPT`: System prompt to pass to the OpenAI model

# COMMAND ----------

# parameters
CHUNK_SIZE = 900
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "text-embedding-3-small"
NUM_CHUNKS = 3
CHAT_MODEL = "gpt-35-turbo-16k"
TEMPERATURE = 0.8
MAX_OUTPUT_TOKENS = 256
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Instantiate Azure OpenAI model clients
# MAGIC
# MAGIC For the LLM and the embedding models

# COMMAND ----------

llm = AzureChatOpenAI(
    deployment_name=CHAT_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_OUTPUT_TOKENS
)

# Remove the OpenAI API base URL from the environment
# because AzureOpenAIEmbeddings gets confused by it
if "OPENAI_API_BASE" in os.environ:
    del os.environ["OPENAI_API_BASE"]

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_MODEL,
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Delta table with parsed document and create vector store

# COMMAND ----------

def create_faiss_database(document_dataframe, content_column, embedding_generator, database_save_directory, chunk_size=500, chunk_overlap=10):
    # Load & split documents
    loader = PySparkDataFrameLoader(spark, document_dataframe, page_content_column=content_column)
    documents = loader.load()
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    document_chunks = document_splitter.split_documents(documents)
    faiss_database = FAISS.from_documents(document_chunks, embedding_generator)

    # Save the FAISS database to the specified directory
    faiss_database.save_local(database_save_directory)

    return faiss_database


def load_retriever(persist_directory):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_MODEL
    )
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True,  # This is required to load the index from MLflow
    )
    return vectorstore.as_retriever()

df_documents = spark.table(f"{SCHEMA}.ai_policy_document").drop("content")

vector_db = create_faiss_database(
  document_dataframe=df_documents,
  content_column="pdf_text",
  embedding_generator=embedding_model,
  database_save_directory=vs_persist_directory,
  chunk_size=CHUNK_SIZE,
  chunk_overlap=CHUNK_OVERLAP
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # Create prompt template
# MAGIC

# COMMAND ----------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load test questions and answers

# COMMAND ----------

df_test_data = pd.read_csv(f"{RAW_FILE_PATH})

# COMMAND ----------

# MAGIC %md
# MAGIC # Define LLM-as-a-judge metric
# MAGIC
# MAGIC You need a GPT-4 deployment for this.

# COMMAND ----------

my_answer_similarity = mlflow.metrics.genai.answer_similarity()

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate model and log experiment

# COMMAND ----------

with mlflow.start_run() as run:
    if "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]


    qa_chain = {"context": vector_db.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    qa_chain.invoke("Testing.")


    model_info = mlflow.langchain.log_model(
        qa_chain,
        artifact_path="retrieval_qa",
        loader_fn=load_retriever,
        persist_dir=vs_persist_directory
        )
    
    df_test_data_ = df_test_data.copy()
    df_test_data_["answer"] = df_test_data_["inputs"].map(lambda x: qa_chain.invoke(x))

    os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_ENDPOINT

    results = mlflow.evaluate(
        data=df_test_data_,
        targets="ground_truth",
        predictions="answer",
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[my_answer_similarity]
        )
    
    mean_answer_similarity = results.tables["eval_results_table"]["answer_similarity/v1/score"].mean()
    mlflow.log_metrics(results.metrics)
    mlflow.log_metric("mean_answer_similarity", mean_answer_similarity)

    mlflow.log_params(
        {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_chunks": NUM_CHUNKS,
            "embedding_model": EMBEDDING_MODEL,
            "chat_model": CHAT_MODEL,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "system_prompt": SYSTEM_PROMPT
        }
    )


    

