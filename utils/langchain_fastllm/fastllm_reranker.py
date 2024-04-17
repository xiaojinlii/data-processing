import os
import sys

import httpx

from typing import Optional, Sequence, Dict, Any

from httpx._types import AuthTypes, HeaderTypes, CookieTypes, VerifyTypes, CertTypes
from langchain_core.documents import Document


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class FastReranker:
    """Document compressor that uses `Cohere Rerank API`."""

    def __init__(
            self,
            url: str,
            top_n: int = 3,
            reranker_score: float = 0,
            *,
            timeout: Optional[float] = None,
            auth: Optional[AuthTypes] = None,
            headers: Optional[HeaderTypes] = None,
            cookies: Optional[CookieTypes] = None,
            verify: VerifyTypes = True,
            cert: Optional[CertTypes] = None,
            client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the client.

        Args:
            url: The url of the server
            timeout: The timeout for requests
            auth: Authentication class for requests
            headers: Headers to send with requests
            cookies: Cookies to send with requests
            verify: Whether to verify SSL certificates
            cert: SSL certificate to use for requests
            client_kwargs: If provided will be unpacked as kwargs to both the sync
                and async httpx clients
        """
        self.top_n = top_n

        _client_kwargs = client_kwargs or {}
        # Enforce trailing slash
        self.url = url if url.endswith("/") else url + "/"
        self.sync_client = httpx.Client(
            base_url=url,
            timeout=timeout,
            auth=auth,
            headers=headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            **_client_kwargs,
        )
        self.async_client = httpx.AsyncClient(
            base_url=url,
            timeout=timeout,
            auth=auth,
            headers=headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            **_client_kwargs,
        )

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            reranker_score: float = 0,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            reranker_score: Scores below reranker_score will be discarded
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []

        doc_list = list(documents)
        texts = [d.page_content for d in doc_list]

        data = {
            "query": query,
            "texts": texts
        }
        response = self.sync_client.post("compute_score_by_query", json=data)
        return self.handle_response(response.json(), doc_list, reranker_score)

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        reranker_score: float = 0,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        if len(documents) == 0:  # to avoid empty api call
            return []

        doc_list = list(documents)
        texts = [d.page_content for d in doc_list]

        data = {
            "query": query,
            "texts": texts
        }
        response = await self.async_client.post("compute_score_by_query", json=data)
        return self.handle_response(response.json(), doc_list, reranker_score)

    def handle_response(self, results, doc_list, reranker_score):
        for index, value in enumerate(results):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value
        doc_list.sort(key=lambda _doc: _doc.metadata["relevance_score"], reverse=True)
        ret_docs = [d for d in doc_list if d.metadata["relevance_score"] > reranker_score]

        top_k = self.top_n if self.top_n < len(ret_docs) else len(ret_docs)
        return ret_docs[:top_k]


async def main():
    reranker_model = FastReranker(url="http://127.0.0.1:9000/reranker", top_n=3)
    docs = [
        Document(page_content="早上好"),
        Document(page_content="哈喽"),
        Document(page_content="嗨"),
        Document(page_content="你好"),
    ]
    result = reranker_model.compress_documents(docs, "哈喽", reranker_score=0.7)
    print(result)
    result = await reranker_model.acompress_documents(docs, "哈喽", reranker_score=0.7)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
