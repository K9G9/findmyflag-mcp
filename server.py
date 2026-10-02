import os
import time
import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from pydantic import BaseModel, ConfigDict
from mcp.types import ToolAnnotations
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

API_URL = "http://127.0.0.1:8002/api/internal/mcp/search"
MAX_IMAGE_BYTES = 30 * 1024 * 1024


mcp = MCPServer(
    name="FindMyFlag",
    description="Visual flag identification using the FindMyFlag index.",
    instructions=(
        "Use FindMyFlag when the user wants to identify, verify, or distinguish "
        "a flag visible in an image. Do not use it for unrelated general questions."
    ),
)



def validate_remote_image_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only HTTPS image URLs are allowed.")

    if parsed.username or parsed.password:
        raise ValueError("Credentials in image URLs are not allowed.")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or 443,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("Image hostname could not be resolved.") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise ValueError("Private or non-public image addresses are not allowed.")


def normalize_result(item: dict) -> dict:
    return {
        "rank": item.get("rank"),
        "score": item.get("score"),
        "title": item.get("title"),
        "subtitle": item.get("subtitle"),
        "country": item.get("country"),
        "organization": item.get("organization"),
        "city": item.get("city"),
        "flag_type": item.get("flag_type"),
        "collection": item.get("collection"),
        "source": item.get("source"),
        "source_label": item.get("source_label"),
        "caption_original": item.get("caption_original"),
        "caption_english": item.get("caption_english"),
        "image_url": item.get("image_url"),
        "source_url": item.get("source_url"),
    }


class ChatGPTFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    download_url: str
    file_id: str
    mime_type: str | None = None
    file_name: str | None = None


@mcp.tool(
    name="identify_flag_from_image",
    description=(
        "Use this when the user wants to identify, verify, or distinguish a flag "
        "visible in an uploaded photo, image, or screenshot. "
        "FindMyFlag specializes in obscure, historical, military, naval, regional, "
        "political, organizational, company, and visually similar flag variants."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    ),
    meta={"openai/fileParams": ["file"]},
)
async def identify_flag_from_image(
    file: ChatGPTFile,
    top_k: int = 10,
) -> dict:
    top_k = max(1, min(int(top_k), 20))

    token = os.environ.get("FMF_INTERNAL_TOKEN")
    if not token:
        return {"error": "FindMyFlag internal authentication is not configured."}

    try:
        image_url = file.download_url
        validate_remote_image_url(image_url)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=3.0),
            follow_redirects=False,
        ) as client:

            image = await client.get(image_url)

            if image.status_code != 200:
                return {"error": "Unable to retrieve the supplied image."}

            content = image.content

            if not content:
                return {"error": "The supplied image was empty."}

            if len(content) > MAX_IMAGE_BYTES:
                return {"error": "Image exceeds the maximum allowed size."}

            content_type = image.headers.get("content-type", "").split(";")[0].lower()

            if not content_type.startswith("image/"):
                return {"error": "The supplied URL did not return an image."}

            response = await client.post(
                API_URL,
                params={"top_k": top_k},
                headers={"X-FMF-Internal-Token": token},
                files={
                    "file": (
                        "image",
                        content,
                        content_type,
                    )
                },
            )

            response.raise_for_status()
            payload = response.json()

    except ValueError:
        return {"error": "The supplied image URL is not allowed."}
    except httpx.HTTPError:
        return {"error": "FindMyFlag search service could not process the image."}

    results = [
        normalize_result(item)
        for item in payload.get("results", [])
    ]

    return {
        "result_count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8010,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        max_request_body_size=4 * 1024 * 1024,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "mcp.findmyflag.com",
                "127.0.0.1:*",
                "localhost:*",
                "[::1]:*",
            ],
            allowed_origins=[
                "https://mcp.findmyflag.com",
                "http://127.0.0.1:*",
                "http://localhost:*",
                "http://[::1]:*",
            ],
        ),
    )
