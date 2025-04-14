# Shoptalk 
## 📖 Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)  
4. [Data Flow](#4-data-flow)  
5. [Deployment Setup](#5-deployment-setup)  
6. [Model Evaluation & Monitoring](#6-model-evaluation--monitoring)  
7. [Dataset](#7-dataset)  
8. [Challenges & Learnings](#8-challenges--learnings)  
9. [Future Improvements](#9-future-improvements) 
10. [Repository Structure](#10-repository-structure)
11. [GoogleColab Notebooks](#11-google-colab-notebooks)
12. [Team Contacts](#12-team-contacts)



## 1. Project Overview 📌

**Objective**:  
The objective of this project is to develop an intelligent AI-powered Shopping Assistant that enables users to interact with an e-commerce platform using natural language queries. Unlike traditional keyword-based search engines, this assistant will understand complex and nuanced queries, retrieve relevant product information using Retrieval-Augmented Generation (RAG), and provide personalized and human-like responses through Natural Language Generation (NLG). The system aims to enhance product discoverability, align search results with user intent, and deliver a superior user experience that is fast, accurate, and conversational.

**Key Features**:

### 1. 🧠 Natural Language Query Understanding
- Supports complex queries like: _“Show me red dresses under $100 for evening wear”_
- Handles synonyms, contextual variations, and user intent

### 2. 🔍 Semantic Product Retrieval (RAG)
- Uses vector-based retrieval + LLMs for highly relevant results
- Searches across product metadata, bullet points, descriptions, etc.

### 3. 🗣️ Natural Language Generation (NLG)
- Converts search results into friendly, coherent responses
- Generates summaries and personalized recommendations

### 4. 🖼️ Multimodal Support (Optional/Future Scope)
- Uses image data and captioning to enhance search accuracy
- Responds to queries like _“something like the blue sneakers I saw yesterday”_

### 5. ⚡ Real-Time Responsiveness
- Fast response time for interactive user experience
- Optimized pipeline for low-latency and scalable performance

### 6. 🧭 User Experience Optimization
- Clear, concise, and helpful recommendations
- Reduces search friction and improves satisfaction
  
## 💼 Business Use Case

Traditional keyword-based search limits user experience and fails to meet modern expectations in e-commerce. This Conversational Shopping Assistant provides:

- ✅ **Higher Conversion Rates**  
  By improving search relevance and matching user intent

- ✅ **Improved Net Promoter Score (NPS)**  
  Natural, intuitive interactions increase customer satisfaction

- ✅ **Lower Bounce Rates**  
  Keeps users engaged with dynamic and personalized responses



## 2. Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Frontend     | Gradio                            |
| Backend      | Airflow / FastAPI / RAG           |
| Libraries    | PyTorch / HuggingFace Transformers|
| Database     | PostgreSQL / Redis / MongoDB      |
| Vector DB    | Chroma / Milvus / FAISS           |
| Deployment   | Docker, EC2                       |
| Evaluation   | MLflow                            |
| LLM & Prompting |	OpenAI GPT, LangChain          | 

## 3. System Architecture

> ![alt text](shoptalk-proj\docs\airflow_dag.png)

## 4. Data Flow

### 📥 User Interaction
- The user inputs a query through the frontend interface.

### 🌐 Frontend Logic
- The frontend captures the input and sends it as an API request to the backend service.

### 🧩 Backend Handling
- The backend receives the request and routes it to the appropriate microservice based on the operation type (e.g., metadata lookup, inference, etc.).

### 🤖 ML Inference
- The inference service performs the following:
  - **Preprocessing** of input data
  - **Embedding Generation** using pre-trained models
  - **Semantic Search** via **Vector DB**
  - **Natural Language Generation** (if applicable) to generate human-readable responses

### 📤 Response
- The result is returned as a structured JSON response.
- The frontend parses the response and displays the output to the user in a friendly format.

## 5. Deployment Details

### 🖥️ Infrastructure

- **EC2 Setup**:  
  - Instance Type: `t2.medium` / `t3.large` (based on compute needs)  
  - Security Groups: Configured and data saved onto S3 bucket 
  - OS: Amazon Linux 2 / Ubuntu 20.04

---

### 🐳 Docker & Microservices

- **Dockerization**:  
  - Each component (frontend, backend, ML service) has its own `Dockerfile`

- **Service Orchestration**:  
  - `docker-compose.yml` to manage multiple containers  
  - (Optional) Kubernetes manifests for production-grade deployment

- **NGINX Reverse Proxy**:  
  - Used for routing requests to appropriate microservices  
  - SSL configured if HTTPS is required

---


## 6. Model Evaluation

### 📈 Metrics Tracked using MLFlow
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### 7. Dataset
- **Details about Dataset**:  
  - Amazon Berkeley Objects (ABO) Dataset. https://amazon-berkeley-objects.s3.amazonaws.com/index.html 
  - Size: [collection of 147,702 product listings with images]
  - Used two main folders- 
    - Product description and metadata - `archives/abo-listings.tar`
    - Catalog imagery, in original and smaller (256px) resolution - `archives/abo-images-small.tar`

## 8. Challenges & Learnings

- Setting up Ec2 with the code thats working on Google colab
- Microservice integration and inter-service communication
- Dockerizing large ML models and optimizing image sizes
- Because we are computing with gpu instead of cpu, our credits was continuosly exhausting if we use entire dataset, so we ran entire dataset using google colab notebook setting it to T4 and on ec2 we had to restrict for 2000 images to continue our project.
- Integrating MLFlow for evaluating RAG and with LLM
- 

---

## 9. Future Improvements

- Handling responses if there are missing images for the given query
- Adding a feedback loop for continuous model improvement
- Enhancing monitoring to include concept drift and data quality alerts

---

## 10. Repository Structure

```bash
shoptalk-proj
├── airflow-dag/              # List of airflow tasks to extract data from jar files and save onto S3
├── rag-ui-service/           # frontend UI to backend RAG communication
├── vector_db_service/        # Searching embeddings using FAISS and Cupy(for GPU)
├── docker-compose.yml        # Compose setup
├── DockerFile/               # configuration for main container
├── docker-compose.yaml/      # configuration file to share other microservices which are inturn running on separte containers
├── README.md                 # This file
└── *.sh                      # created shell cripts to cleanup, settingup environment, pull secerets and keys and start docker container
```

## 11. GoogleColab Notebooks

  - Links to Google colab notebooks that we worked 
  
## 12. Team Contacts

- **Authors**: Sagar Vasamsetty, Vijaya Lakshmi Ijjada, Sumit Kumar Singh, Srinivas Nagamantha  
- **Email**: vasamsetty@gmail.com,onlinevijaya30@gmail.com,kumarsumitsingh@gmail.com,nagamantha11@gmail.com 
- **GitHub**: https://github.com/sagarsGitArena/shop-talk/tree/main/shoptalk-proj



