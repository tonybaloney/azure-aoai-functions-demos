import azure.functions as func
import json
import azure.identity
import openai
import os

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="text_to_embedding")
def text_to_embedding(req: func.HttpRequest) -> func.HttpResponse:
    """
    Convert a text input into an embedding using the Azure OpenAI service.
    """
    text = req.params.get('text')
    if not text:
        raise ValueError("Please pass a text in the query string")

    AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
    AZURE_OPENAI_ADA_DEPLOYMENT = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT")

    # When running locally, use VS Code auth as the default credential by installing the 
    # Azure VS Code extensions and logging in.
    # When running in prod, use a Managed Identity for Azure Functions -> Azure OpenAI
    azure_credential = azure.identity.DefaultAzureCredential()
    token_provider = azure.identity.get_bearer_token_provider(azure_credential,
        "https://cognitiveservices.azure.com/.default")

    openai_client = openai.AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
        azure_ad_token_provider=token_provider)

    get_embeddings_response = openai_client.embeddings.create(model=AZURE_OPENAI_ADA_DEPLOYMENT, input=text)
    embedding = get_embeddings_response.data[0].embedding

    return func.HttpResponse(
        json.dumps(embedding),
        mimetype="application/json",
        status_code=200
    )

