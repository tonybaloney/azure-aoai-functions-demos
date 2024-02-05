from typing import Sequence
import azure.functions as func
import json
import azure.identity
import openai
import os
import numpy as np

AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_ADA_DEPLOYMENT = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT")


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def get_client():
    # When running locally, use VS Code auth as the default credential by installing the 
    # Azure VS Code extensions and logging in.
    # When running in prod, use a Managed Identity for Azure Functions -> Azure OpenAI
    azure_credential = azure.identity.DefaultAzureCredential()
    token_provider = azure.identity.get_bearer_token_provider(azure_credential,
        "https://cognitiveservices.azure.com/.default")

    return openai.AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
        azure_ad_token_provider=token_provider)

def cosine_similarity(a: Sequence[float], b: Sequence[float]):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route(route="text_to_embedding")
def text_to_embedding(req: func.HttpRequest) -> func.HttpResponse:
    """
    Convert a text input into an embedding using the Azure OpenAI service.
    """
    text = req.params.get('text')
    if not text:
        raise ValueError("Please pass a text in the query string")
    openai_client = get_client()

    get_embeddings_response = openai_client.embeddings.create(model=AZURE_OPENAI_ADA_DEPLOYMENT, input=text)
    embedding = get_embeddings_response.data[0].embedding

    return func.HttpResponse(
        json.dumps(embedding),
        mimetype="application/json",
        status_code=200
    )


@app.route("compare_sentences")
def compare_sentences(req: func.HttpRequest) -> func.HttpResponse:
    """
    Compare two sentences and return a similarity score.
    """
    sentence1 = req.params.get('sentence1')
    sentence2 = req.params.get('sentence2')
    if not sentence1 or not sentence2:
        raise ValueError("Please pass sentence1 and sentence2 in the query string")
    openai_client = get_client()

    sentences_response = openai_client.embeddings.create(model=AZURE_OPENAI_ADA_DEPLOYMENT, input=[sentence1, sentence2])
    similarity_score = cosine_similarity(sentences_response.data[0].embedding, sentences_response.data[1].embedding)

    return func.HttpResponse(
        json.dumps(similarity_score),
        mimetype="application/json",
        status_code=200
    )


@app.route("movie_search")
def movie_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    Search for a movie using the Azure OpenAI service.
    """
    movie_title = req.params.get('movie_title')
    n = int(req.params.get('n', 10))
    if not movie_title:
        raise ValueError("Please pass a movie_title in the query string")
    openai_client = get_client()

    # Load in vectors for movie titles
    with open('data/openai_movies.json', encoding='utf-8') as json_file:
        movie_vectors = json.load(json_file)

    embeddings_response = openai_client.embeddings.create(model=AZURE_OPENAI_ADA_DEPLOYMENT, input=movie_title)
    vector = embeddings_response.data[0].embedding

    # Compute cosine similarity between query and each movie title
    scores = []
    for movie in movie_vectors:
        scores.append((movie, cosine_similarity(vector, movie_vectors[movie])))
    
    # Sort by similarity score
    scores.sort(key=lambda x: x[1], reverse=True)
    # Return top 10 as JSON
    return func.HttpResponse(
        json.dumps(scores[:n]),
        mimetype="application/json",
        status_code=200
    )

