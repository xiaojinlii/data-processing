from typing import List, Optional, Dict, Any

import httpx
from httpx._types import AuthTypes, HeaderTypes, CookieTypes, VerifyTypes, CertTypes
from langchain_core.embeddings import Embeddings


class FastEmbeddings(Embeddings):

    def __init__(
        self,
        url: str,
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        response = self.sync_client.post("embed_documents", json=texts)
        result = response.json()
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = self.sync_client.post("embed_query", json=text)
        result = response.json()
        return result

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        response = await self.async_client.post("embed_documents", json=texts)
        result = response.json()
        return result

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        response = await self.async_client.post("embed_query", json=text)
        result = response.json()
        return result


async def main():
    embeddings = FastEmbeddings(url="http://127.0.0.1:9000/embeddings")
    print(embeddings.embed_query("哈喽"))
    print(embeddings.embed_documents(["哈喽", "嗨"]))
    print(await embeddings.aembed_query("哈喽"))
    print(await embeddings.aembed_documents(["哈喽", "嗨"]))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
