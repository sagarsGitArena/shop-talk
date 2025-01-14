import os
import tarfile
import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import shutil
import gzip
import glob
import json
from utils.json_utils import flatten_json
from utils.s3_utils import upload_file_to_s3
import re
import numpy as np
#import mlflow
#from sentence_transformers import SentenceTransformer

from datetime import datetime, timedelta


from config import LISTINGS_DOWNLOAD_PATH_URL, LOCAL_RAW_DATA_DIR, ALL_LISTINGS_DATA_CSV, US_ONLY_LISTINGS_CSV, US_PRODUCT_IMAGE_MERGE_CSV, AWS_S3_BUCKET, LISTINGS_CSV_FILE_LOCATION, IMAGES_DOWNLOAD_PATH_URL,LOCAL_RAW_IMGS_DIR, IMAGES_CSV_FILE_LOCATION, IMAGES_CSV_FILE, TMP_LISTINGS_SOURCE, TAR_FILE_NAME
from tasks.definitions import download_tar_file, extract_tar_file, flatten_each_json_and_save_as_csv, flatten_all_json_and_save_as_csv, perform_eda_on_us_listings_data, flatten_to_csv_images, download_tar_file_images, extract_tar_file_images, up_load_us_listings_to_s3, merge_listings_images, copy_listings_tar_file


# DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="shoptalk_ingestion_pipeline",
    default_args=default_args,
    description="Pipeline to download, extract, and process product listings",
    start_date=datetime(2024, 1, 1),
    #schedule_interval="@daily",
    #schedule_interval="*/10 * * * *",  # Every 10 minutes
    schedule_interval=timedelta(minutes=10),  # Every 10 minutes
    max_active_runs=1,
    catchup=False,
) as dag:

## TEmporarily disable download task instead have copy task

    download_task = PythonOperator(
        task_id="download_tar_file",
        python_callable=download_tar_file,
        dag=dag
    )

    extract_task = PythonOperator(
        task_id="extract_tar_file",
        python_callable=extract_tar_file,
        op_kwargs={ 
                    "extract_dir_path": LOCAL_RAW_DATA_DIR, 
                    "tar_file": "abo-listings.tar", 
                    "local_extracted_json_dir": "listings/metadata/", 
                    "extracted_file_pattern": "listings_?.json.gz",
                    "decompressed_json_file_pattern": "listings_?.json"
                },
        #provide_context=True,        
        trigger_rule='all_success',
        depends_on_past=False,
        dag=dag
        
    )

    copy_listings_task = PythonOperator(
        task_id="copy_listings_tar_file",
        python_callable=copy_listings_tar_file,
        op_kwargs={
                    "source_path": TMP_LISTINGS_SOURCE,
                    "destination_path": LOCAL_RAW_DATA_DIR,
                    "tar_file_name" : TAR_FILE_NAME
                   },
        trigger_rule='all_success',
        depends_on_past=False,
        dag=dag
    )
    

    
    flatten_all_json_and_save_as_csv = PythonOperator(
        task_id="flatten_all_json_and_save_as_csv",
        python_callable=flatten_all_json_and_save_as_csv,
        op_kwargs= {"local_extracted_json_dir": "listings/metadata/"                   
        },
        #provide_context=True,        
        trigger_rule='all_success',
        depends_on_past=False,
        dag=dag
    )



  # Task 1: Download the images tar file
    download_images_task = PythonOperator(
        task_id="download_tar_file_images",
        python_callable=download_tar_file_images,
        depends_on_past=False,
        dag=dag
    )

    # Task 2: Extract the images tar file
    extract_images_task = PythonOperator(
        task_id="extract_tar_file_images",
        python_callable=extract_tar_file_images,
        trigger_rule='all_success',
        depends_on_past=False,
        dag=dag
    )

    # Task 3: Flatten the image metadata to a CSV file
    flatten_images_metadata_task = PythonOperator(
        task_id="flatten_to_csv_images",
        python_callable=flatten_to_csv_images,
        trigger_rule='all_success',
        depends_on_past=False,
        dag=dag
    )
    # Define task dependencies

    merge_listings_image_df_task = PythonOperator(
            task_id="merge_listings_images",
            python_callable=merge_listings_images,
            trigger_rule='all_success',
            depends_on_past=False,
            dag=dag
    )
    # [download_task >> extract_task >> flatten_all_json_and_save_as_csv >>upload_listings_to_s3, download_images_task >> extract_images_task >> flatten_images_metadata_task] >> merge_listings_image_df_task
## If we are downloading and extracting the tar
#[download_task >> extract_task >> flatten_all_json_and_save_as_csv , download_images_task >> extract_images_task >> flatten_images_metadata_task] >> merge_listings_image_df_task

## If we are copying the tar file from local dir for minimal dataset
[copy_listings_task >> extract_task >> flatten_all_json_and_save_as_csv , download_images_task >> extract_images_task >> flatten_images_metadata_task] >> merge_listings_image_df_task

