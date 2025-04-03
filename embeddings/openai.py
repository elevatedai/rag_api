"""OpenAI Embeddings wrapper for Langchain with custom base URL support."""

import base64
from os.path import exists
from typing import Any, Dict, List, Literal, Optional, Union, Self
from urllib.parse import urlparse

import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, model_validator, SecretStr

OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "text-embedding-3-small"
VALID_ENCODING_FORMATS = ["float", "base64"]
DEFAULT_BATCH_SIZE = 1000


def is_local(url: str) -> bool:
    """Check if a URL is a local file."""
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


def get_bytes_str(file_path: str) -> str:
    """Get the bytes string of a file."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def secret_from_env(env_name: str, default=None):
    """Get a SecretStr from an environment variable."""
    import os
    
    def get_secret():
        value = os.getenv(env_name, default)
        if value is None:
            return None
        return SecretStr(value)
    
    return get_secret


class OpenAIEmbeddings(BaseModel, Embeddings):
    """OpenAI embedding models with custom base URL support."""

    client: Optional[Any] = None  # :meta private:
    model: Optional[str] = DEFAULT_MODEL
    openai_api_key: Optional[SecretStr] = Field(default_factory=secret_from_env("OPENAI_API_KEY", default=None))
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    openai_api_base: Optional[str] = OPENAI_API_BASE
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that auth token exists in environment."""
        # Check if API key is present
        if self.openai_api_key is None:
            raise ValueError(
                "OpenAI API key is required. Please set the OPENAI_API_KEY environment "
                "variable or pass it directly as openai_api_key."
            )

        # Validate encoding format
        if self.encoding_format not in VALID_ENCODING_FORMATS:
            raise ValueError(
                f"Encoding format {self.encoding_format} not supported. "
                f"Choose from {VALID_ENCODING_FORMATS}"
            )

        # Validate batch size
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Create a session with headers
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.openai_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }
        )
        self.client = session
        return self

    def _get_embedding_url(self) -> str:
        """Get the appropriate embedding URL based on settings."""
        return f"{self.openai_api_base.rstrip('/')}/embeddings"

    def _process_embeddings(self, response: Dict) -> List[List[float]]:
        """Process embeddings from API response."""
        if "data" not in response:
            raise RuntimeError(response.get("error", {}).get("message", "Unknown error occurred"))
        
        embeddings = []
        for item in sorted(response["data"], key=lambda e: e["index"]):
            if self.encoding_format == "base64" and isinstance(item["embedding"], str):
                # Decode base64 to floats if necessary
                decoded_bytes = base64.b64decode(item["embedding"])
                embedding = np.frombuffer(decoded_bytes, dtype=np.float32).tolist()
                embeddings.append(embedding)
            else:
                embeddings.append(item["embedding"])
                
        return embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": self.encoding_format,
        }

        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        response = self.client.post(self._get_embedding_url(), json=payload).json()
        return self._process_embeddings(response)

    def _embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Internal method to get embeddings with batching support."""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Process in batches
        all_embeddings = []
        batch_size = self.batch_size or DEFAULT_BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query text."""
        return self._embed([text])[0]

    def _process_image_input(self, uris: List[str]) -> List[Dict[str, str]]:
        """Process image URIs into the format expected by the API."""
        input_data = []
        for uri in uris:
            if is_local(uri):
                input_data.append({"image": get_bytes_str(uri)})
            else:
                input_data.append({"image": uri})
        return input_data
        
    def _embed_image_batch(self, image_inputs: List[Dict[str, str]]) -> List[List[float]]:
        """Embed a batch of images."""
        payload = {
            "input": image_inputs,
            "model": self.model,
            "encoding_format": self.encoding_format,
        }
        
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
            
        response = self.client.post(self._get_embedding_url(), json=payload).json()
        return self._process_embeddings(response)

    def embed_image(self, uri: str) -> List[float]:
        """Get embedding for a single image."""
        image_inputs = self._process_image_input([uri])
        return self._embed_image_batch(image_inputs)[0]

    def embed_images(self, uris: List[str]) -> List[List[float]]:
        """Get embeddings for multiple images."""
        # Process in batches
        all_embeddings = []
        image_inputs = self._process_image_input(uris)
        batch_size = self.batch_size or DEFAULT_BATCH_SIZE
        
        for i in range(0, len(image_inputs), batch_size):
            batch_inputs = image_inputs[i:i + batch_size]
            batch_embeddings = self._embed_image_batch(batch_inputs)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings