# Databricks notebook source
# MAGIC %pip install pypdf==4.1.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import io
import re
import pyspark.sql.functions as F
from typing import Iterator
import pandas as pd

# COMMAND ----------

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)


# COMMAND ----------

SCHEMA = "your-schema-name" # your schema name
VOLUME_PATH = f"dbfs:/path/to/your/directory/with/the/pdf" # path to your directory where your PDF files are

# COMMAND ----------

# MAGIC %md
# MAGIC # Ingest Raw Binaries

# COMMAND ----------

df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load('dbfs:'+VOLUME_PATH))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{VOLUME_PATH}/checkpoints/raw_docs')
  .table(f'{SCHEMA}.pdf_raw').awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC Parse PDFs

# COMMAND ----------

import warnings
from pypdf import PdfReader

def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        parsed_content = [page_content.extract_text() for page_content in reader.pages]
        return "\n".join(parsed_content)
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# COMMAND ----------

@F.pandas_udf("string")
def read_pdf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    import warnings
    from pypdf import PdfReader

    def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
        try:
            pdf = io.BytesIO(raw_doc_contents_bytes)
            reader = PdfReader(pdf)
            parsed_content = [page_content.extract_text() for page_content in reader.pages]
            return "\n".join(parsed_content)
        except Exception as e:
            warnings.warn(f"Exception {e} has been thrown during parsing")
            return None
          
    for x in batch_iter:
        yield x.apply(parse_bytes_pypdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Parse PDF

# COMMAND ----------

df_pdfs = spark.read.table(f"{SCHEMA}.pdf_raw")

# COMMAND ----------

df_pdfs = df_pdfs.withColumn("pdf_text", read_pdf("content"))

# COMMAND ----------

df_pdfs.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Write Delta table

# COMMAND ----------

df_pdfs.write.mode("overwrite").saveAsTable(f"{SCHEMA}.ai_policy_document")

# COMMAND ----------


