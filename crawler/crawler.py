# /// script
# dependencies = [
#     "crawl4ai>=0.4.0",
#     "rich>=13.7.0",
#     "pydantic>=2.0.0",
#     "requests>=2.31.0",
# ]
# ///

"""Web Crawler Agent - Extract and structure content from websites with multiple crawling strategies."""

import argparse
import asyncio
import re
import sys
import xml.etree.ElementTree as ET
from enum import Enum
from itertools import chain
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel

console = Console()


class CrawlStrategy(Enum):
    """Available crawling strategies."""

    SINGLE = "single"
    RECURSIVE = "recursive"
    SITEMAP = "sitemap"
    MARKDOWN = "markdown"


class ContentChunk(BaseModel):
    """Individual content chunk with metadata."""

    text: str = Field(description="Chunk content")
    headers: List[str] = Field(default_factory=list, description="Extracted headers")
    position: int = Field(description="Chunk position in document")
    char_count: int = Field(description="Character count")
    word_count: int = Field(description="Word count")

    @field_validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v.strip()


class CrawlResult(BaseModel):
    """Complete crawl result for a single URL."""

    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Page title")
    markdown: str = Field(description="Extracted markdown content")
    chunks: List[ContentChunk] = Field(
        default_factory=list, description="Content chunks"
    )
    success: bool = Field(description="Crawl success status")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CrawlStats(BaseModel):
    """Crawling session statistics."""

    total_urls: int = Field(description="Total URLs processed")
    successful: int = Field(description="Successfully crawled URLs")
    failed: int = Field(description="Failed crawl attempts")
    total_chunks: int = Field(description="Total content chunks generated")
    processing_time: float = Field(description="Total processing time in seconds")


def detect_crawl_strategy(url: str) -> CrawlStrategy:
    """Detect the appropriate crawling strategy based on URL patterns."""
    url_lower = url.lower()
    if url_lower.endswith("sitemap.xml") or "sitemap" in url_lower:
        return CrawlStrategy.SITEMAP
    elif url_lower.endswith((".txt", ".md")):
        return CrawlStrategy.MARKDOWN
    return CrawlStrategy.SINGLE


def normalize_url(url: str) -> str:
    """Remove URL fragments for deduplication."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{parsed.params}"


def extract_sitemap_urls(sitemap_url: str) -> List[str]:
    """Extract URLs from XML sitemap."""
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"

        root = ET.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        return [
            loc.text
            for loc in root.findall(".//ns:loc", namespace)
            if loc.text and loc.text.startswith("http")
        ]
    except Exception as e:
        console.print(f"[red]Failed to parse sitemap {sitemap_url}: {e}[/red]")
        return []


def split_by_headers(markdown: str, max_chunk_size: int = 1000) -> List[str]:
    """Split markdown content by headers hierarchically."""
    if len(markdown) <= max_chunk_size:
        return [markdown]

    def split_by_pattern(text: str, pattern: str) -> List[str]:
        """Split text by regex pattern, keeping delimiters."""
        indices = [m.start() for m in re.finditer(pattern, text, re.MULTILINE)]
        if not indices:
            return [text]

        indices.append(len(text))
        return [
            text[indices[i] : indices[i + 1]].strip()
            for i in range(len(indices) - 1)
            if text[indices[i] : indices[i + 1]].strip()
        ]

    def split_large_chunks(chunks: List[str], max_size: int) -> List[str]:
        """Further split chunks that exceed max size."""
        result = []
        for chunk in chunks:
            if len(chunk) <= max_size:
                result.append(chunk)
            else:
                words = chunk.split()
                current_chunk = []
                current_size = 0

                for word in words:
                    word_size = len(word) + 1
                    if current_size + word_size > max_size and current_chunk:
                        result.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_size = word_size
                    else:
                        current_chunk.append(word)
                        current_size += word_size

                if current_chunk:
                    result.append(" ".join(current_chunk))

        return result

    # Split by headers in order of priority
    patterns = [r"^# .+$", r"^## .+$", r"^### .+$"]
    chunks = [markdown]

    for pattern in patterns:
        chunks = list(
            chain.from_iterable(split_by_pattern(chunk, pattern) for chunk in chunks)
        )
        if all(len(chunk) <= max_chunk_size for chunk in chunks):
            break

    return split_large_chunks(chunks, max_chunk_size)


def extract_chunk_headers(text: str) -> List[str]:
    """Extract header text from markdown chunk."""
    header_pattern = re.compile(r"^(#+)\s+(.+)$", re.MULTILINE)
    return [match.group(2).strip() for match in header_pattern.finditer(text)]


def sanitize_text(text: str) -> str:
    """Clean text of problematic characters and ensure valid UTF-8."""
    if not text:
        return ""

    # Replace common problematic characters
    text = text.replace("\ufffd", "")  # Remove replacement characters
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", "", text
    )  # Remove control chars

    # Ensure valid UTF-8 encoding
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="ignore").decode("utf-8")


def create_content_chunk(text: str, position: int) -> ContentChunk:
    """Create a ContentChunk with computed metadata."""
    clean_text = sanitize_text(text)
    headers = extract_chunk_headers(clean_text)
    words = clean_text.split()

    return ContentChunk(
        text=clean_text,
        headers=headers,
        position=position,
        char_count=len(clean_text),
        word_count=len(words),
    )


def process_markdown_content(
    markdown: str, max_chunk_size: int = 1000
) -> List[ContentChunk]:
    """Process markdown into structured chunks."""
    if not markdown or not markdown.strip():
        return []

    # Sanitize the entire markdown first
    clean_markdown = sanitize_text(markdown)
    if not clean_markdown.strip():
        return []

    raw_chunks = split_by_headers(clean_markdown, max_chunk_size)
    return [
        create_content_chunk(chunk, i)
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]


async def crawl_single_url(
    crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig
) -> CrawlResult:
    """Crawl a single URL and extract content."""
    try:
        result = await crawler.arun(url=url, config=config)

        if not result.success:
            return CrawlResult(
                url=url,
                markdown="",
                success=False,
                error=f"Crawl failed: {result.error_message or 'Unknown error'}",
            )

        chunks = process_markdown_content(result.markdown or "")

        # Sanitize title and markdown
        title = sanitize_text(getattr(result, "title", "") or "")
        markdown = sanitize_text(result.markdown or "")

        return CrawlResult(
            url=url,
            title=title,
            markdown=markdown,
            chunks=chunks,
            success=True,
            metadata={
                "links_count": len(getattr(result, "links", {}).get("internal", [])),
                "images_count": len(getattr(result, "media", {}).get("images", [])),
            },
        )

    except Exception as e:
        return CrawlResult(url=url, markdown="", success=False, error=str(e))


async def crawl_recursive(
    start_url: str, max_depth: int = 2, max_concurrent: int = 5
) -> List[CrawlResult]:
    """Recursively crawl a website following internal links."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=NoExtractionStrategy(),
        stream=False,
    )

    visited = set()
    current_urls = {normalize_url(start_url)}
    all_results = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            if not current_urls:
                break

            urls_to_crawl = [
                url for url in current_urls if normalize_url(url) not in visited
            ]

            if not urls_to_crawl:
                break

            console.print(
                f"[cyan]Depth {depth + 1}: Crawling {len(urls_to_crawl)} URLs[/cyan]"
            )

            # Crawl current level URLs
            crawl_tasks = [
                crawl_single_url(crawler, url, config) for url in urls_to_crawl
            ]

            batch_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)

            next_level_urls = set()

            for result in batch_results:
                if isinstance(result, Exception):
                    console.print(f"[red]Error: {result}[/red]")
                    continue

                if not isinstance(result, CrawlResult):
                    continue

                all_results.append(result)
                visited.add(normalize_url(result.url))

                if result.success and depth < max_depth - 1:
                    # Extract internal links for next level
                    try:
                        crawl_result = await crawler.arun(url=result.url, config=config)
                        if (
                            hasattr(crawl_result, "links")
                            and "internal" in crawl_result.links
                        ):
                            for link in crawl_result.links["internal"]:
                                if "href" in link:
                                    next_url = normalize_url(link["href"])
                                    if next_url not in visited:
                                        next_level_urls.add(next_url)
                    except Exception:
                        pass  # Skip link extraction errors

            current_urls = next_level_urls

    return all_results


async def crawl_sitemap_urls(
    urls: List[str], max_concurrent: int = 10
) -> List[CrawlResult]:
    """Crawl multiple URLs in parallel from sitemap."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=NoExtractionStrategy(),
        stream=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process in batches to avoid overwhelming the system
        batch_size = max_concurrent
        all_results = []

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            console.print(
                f"[cyan]Processing batch {i // batch_size + 1}: {len(batch_urls)} URLs[/cyan]"
            )

            tasks = [crawl_single_url(crawler, url, config) for url in batch_urls]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, CrawlResult):
                    all_results.append(result)
                elif isinstance(result, Exception):
                    console.print(f"[red]Batch error: {result}[/red]")

        return all_results


def save_results(results: List[CrawlResult], output_path: str) -> None:
    """Save crawl results to XML file."""
    stats = calculate_stats(results)

    # Create root element
    root = ET.Element("crawl_results")

    # Add stats as attributes
    stats_elem = ET.SubElement(root, "stats")
    stats_elem.set("total_urls", str(stats.total_urls))
    stats_elem.set("successful", str(stats.successful))
    stats_elem.set("failed", str(stats.failed))
    stats_elem.set("total_chunks", str(stats.total_chunks))
    stats_elem.set("processing_time", str(stats.processing_time))

    # Add results
    for result in results:
        result_elem = ET.SubElement(root, "result")
        result_elem.set("url", result.url)
        result_elem.set("success", str(result.success).lower())

        # Add title
        if result.title:
            title_elem = ET.SubElement(result_elem, "title")
            title_elem.text = result.title

        # Add error if present
        if result.error:
            error_elem = ET.SubElement(result_elem, "error")
            error_elem.text = result.error

        # Add metadata
        if result.metadata:
            metadata_elem = ET.SubElement(result_elem, "metadata")
            for key, value in result.metadata.items():
                metadata_elem.set(key, str(value))

        # Add markdown content
        if result.markdown:
            markdown_elem = ET.SubElement(result_elem, "markdown")
            markdown_elem.text = result.markdown

        # Add chunks
        for chunk in result.chunks:
            chunk_elem = ET.SubElement(result_elem, "chunk")
            chunk_elem.set("position", str(chunk.position))
            chunk_elem.set("char_count", str(chunk.char_count))
            chunk_elem.set("word_count", str(chunk.word_count))

            # Add headers
            if chunk.headers:
                headers_elem = ET.SubElement(chunk_elem, "headers")
                for header in chunk.headers:
                    header_elem = ET.SubElement(headers_elem, "header")
                    header_elem.text = header

            # Add text content
            text_elem = ET.SubElement(chunk_elem, "text")
            text_elem.text = chunk.text

    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)  # Pretty print

    try:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        console.print(f"[green]Results saved to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving XML: {e}[/red]")
        raise


def calculate_stats(results: List[CrawlResult]) -> CrawlStats:
    """Calculate session statistics."""
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(len(r.chunks) for r in results)

    return CrawlStats(
        total_urls=len(results),
        successful=successful,
        failed=len(results) - successful,
        total_chunks=total_chunks,
        processing_time=0.0,  # Would need timing implementation
    )


def display_results(results: List[CrawlResult], show_content: bool = False) -> None:
    """Display crawl results with rich formatting."""
    stats = calculate_stats(results)

    # Summary panel
    summary = f"""[bold]URLs Processed:[/bold] {stats.total_urls}
[bold]Successful:[/bold] [green]{stats.successful}[/green]
[bold]Failed:[/bold] [red]{stats.failed}[/red]
[bold]Total Chunks:[/bold] {stats.total_chunks}"""

    console.print(Panel(summary, title="📊 Crawl Summary", border_style="blue"))

    # Results details
    for i, result in enumerate(results, 1):
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        title = result.title[:50] + "..." if len(result.title) > 50 else result.title

        result_text = f"{status} {result.url}"
        if title:
            result_text += f"\n  Title: {title}"
        if result.success:
            result_text += f"\n  Chunks: {len(result.chunks)}"
        else:
            result_text += f"\n  Error: {result.error}"

        if show_content and result.success and result.chunks:
            result_text += f"\n  Preview: {result.chunks[0].text[:100]}..."

        console.print(f"[dim]{i}.[/dim] {result_text}")


async def main():
    parser = argparse.ArgumentParser(
        description="Web Crawler Agent - Extract structured content from websites",
        epilog="Example: uv run crawler.py https://example.com --recursive --depth 2",
    )

    parser.add_argument("url", help="URL to crawl (website, sitemap, or markdown file)")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in CrawlStrategy],
        help="Force specific crawling strategy (auto-detected if not specified)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively crawl internal links",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=2,
        help="Maximum crawl depth for recursive strategy (default: 2)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=5,
        help="Maximum parallel requests (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--output", "-o", help="Output XML file path (default: auto-generated)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )
    parser.add_argument(
        "--show-content", action="store_true", help="Show content previews in results"
    )

    args = parser.parse_args()

    # Determine strategy
    if args.recursive:
        strategy = CrawlStrategy.RECURSIVE
    elif args.strategy:
        strategy = CrawlStrategy(args.strategy)
    else:
        strategy = detect_crawl_strategy(args.url)

    if not args.quiet:
        console.print(f"[cyan]Using strategy: {strategy.value}[/cyan]")

    try:
        # Execute crawling based on strategy
        if strategy == CrawlStrategy.SITEMAP:
            sitemap_urls = extract_sitemap_urls(args.url)
            if not sitemap_urls:
                console.print("[red]No URLs found in sitemap[/red]")
                return 1

            if not args.quiet:
                console.print(f"[cyan]Found {len(sitemap_urls)} URLs in sitemap[/cyan]")

            results = await crawl_sitemap_urls(sitemap_urls, args.parallel)

        elif strategy == CrawlStrategy.RECURSIVE:
            results = await crawl_recursive(args.url, args.depth, args.parallel)

        elif strategy == CrawlStrategy.MARKDOWN:
            browser_config = BrowserConfig(headless=True)
            config = CrawlerRunConfig(extraction_strategy=NoExtractionStrategy())

            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = [await crawl_single_url(crawler, args.url, config)]

        else:  # SINGLE
            browser_config = BrowserConfig(headless=True)
            config = CrawlerRunConfig(extraction_strategy=NoExtractionStrategy())

            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = [await crawl_single_url(crawler, args.url, config)]

        # Process results
        if not results:
            console.print("[red]No results generated[/red]")
            return 1

        # Save output
        if args.output:
            output_path = args.output
        else:
            domain = urlparse(args.url).netloc or "crawl"
            output_path = f"{domain}_{strategy.value}_results.xml"

        save_results(results, output_path)

        # Display results
        if not args.quiet:
            display_results(results, args.show_content)

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Crawling interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        console.print(
            Panel(
                "[bold]Web Crawler Agent[/bold]\n\n"
                "Extract and structure content from websites\n\n"
                "[yellow]Usage:[/yellow]\n"
                "  uv run crawler.py https://example.com\n"
                "  uv run crawler.py https://site.com --recursive --depth 3\n"
                "  uv run crawler.py https://site.com/sitemap.xml --parallel 10\n"
                "  uv run crawler.py https://site.com/file.txt\n\n"
                "[yellow]Strategies:[/yellow]\n"
                "  • Single page extraction\n"
                "  • Recursive site crawling\n"
                "  • Sitemap batch processing\n"
                "  • Markdown/text file processing",
                title="🕷️ Web Crawler Agent",
                border_style="cyan",
            )
        )
        sys.exit(0)
    else:
        sys.exit(asyncio.run(main()))
