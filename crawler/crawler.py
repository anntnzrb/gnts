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
import functools
import re
import sys
import time
import xml.etree.ElementTree as ET
from enum import Enum
from itertools import chain
from operator import attrgetter
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table

console = Console()

get_processing_time = attrgetter("metadata.processing_time")
is_successful = attrgetter("success")


def get_chunk_count(result):
    """Get chunk count from result."""
    return len(result.chunks)


def get_char_count(result):
    """Get total character count from result chunks."""
    return sum(chunk.char_count for chunk in result.chunks)


def format_size(size):
    """Format size in readable format."""
    return f"{size // 1024}K" if size > 1024 else str(size)


def sanitize_filename(text):
    """Sanitize text for filename use."""
    return re.sub(r'[<>:"/\\|?*]', "_", text)


# Progress bar factory
def create_progress() -> Progress:
    """Create standardized progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )


def pipe(*functions):
    """Compose functions left-to-right (pipe)."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def safe_get(dictionary, key, default=None):
    """Safe dictionary access with default."""
    return dictionary.get(key, default) if dictionary else default


def partition(predicate, iterable):
    """Partition entries into false entries and true entries."""
    from itertools import filterfalse, tee

    iterable1, iterable2 = tee(iterable)
    return filterfalse(predicate, iterable1), filter(predicate, iterable2)


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
        error_panel = Panel(
            f"[red]Failed to parse sitemap:[/red]\n[cyan]{sitemap_url}[/cyan]\n[red]Error: {str(e)}[/red]",
            title="[red]❌ Sitemap Error[/red]",
            border_style="red",
            padding=(0, 1),
        )
        console.print(error_panel)
        return []


def split_by_headers(markdown: str, max_chunk_size: int = 1000) -> List[str]:
    """Split markdown content by headers hierarchically."""
    if len(markdown) <= max_chunk_size:
        return [markdown]

    def split_by_pattern(text: str, pattern: str) -> List[str]:
        """Split text by regex pattern, keeping delimiters."""
        indices = [m.start() for m in re.finditer(pattern, text, re.MULTILINE)] + [
            len(text)
        ]
        return (
            [
                text[indices[i] : indices[i + 1]].strip()
                for i in range(len(indices) - 1)
                if text[indices[i] : indices[i + 1]].strip()
            ]
            if indices[:-1]
            else [text]
        )

    def split_large_chunks(chunks: List[str], max_size: int) -> List[str]:
        """Further split chunks that exceed max size."""

        def split_chunk(chunk: str) -> List[str]:
            if len(chunk) <= max_size:
                return [chunk]

            words = chunk.split()
            result, current_chunk, current_size = [], [], 0

            for word in words:
                word_size = len(word) + 1
                if current_size + word_size > max_size and current_chunk:
                    result.append(" ".join(current_chunk))
                    current_chunk, current_size = [word], word_size
                else:
                    current_chunk.append(word)
                    current_size += word_size

            return result + ([" ".join(current_chunk)] if current_chunk else [])

        return list(chain.from_iterable(map(split_chunk, chunks)))

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

    # text cleaning pipeline
    clean_pipeline = pipe(
        lambda t: t.replace("\ufffd", ""),
        lambda t: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", "", t),
        lambda t: t.encode("utf-8", errors="ignore").decode("utf-8"),
    )

    return clean_pipeline(text)


def create_content_chunk(text: str, position: int) -> ContentChunk:
    """Create a ContentChunk with computed metadata."""
    clean_text = sanitize_text(text)
    return ContentChunk(
        text=clean_text,
        headers=extract_chunk_headers(clean_text),
        position=position,
        char_count=len(clean_text),
        word_count=len(clean_text.split()),
    )


def process_markdown_content(
    markdown: str, max_chunk_size: int = 1000
) -> List[ContentChunk]:
    """Process markdown into structured chunks."""
    if not markdown or not markdown.strip():
        return []

    clean_markdown = sanitize_text(markdown)
    if not clean_markdown.strip():
        return []

    return [
        create_content_chunk(chunk, i)
        for i, chunk in enumerate(split_by_headers(clean_markdown, max_chunk_size))
        if chunk.strip()
    ]


async def crawl_single_url(
    crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig
) -> CrawlResult:
    """Crawl a single URL and extract content."""
    start_time = time.time()
    try:
        result = await crawler.arun(url=url, config=config)
        processing_time = time.time() - start_time

        if not result.success:
            return CrawlResult(
                url=url,
                markdown="",
                success=False,
                error=f"Crawl failed: {result.error_message or 'Unknown error'}",
                metadata={"processing_time": processing_time},
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
                "processing_time": processing_time,
            },
        )

    except Exception as e:
        processing_time = time.time() - start_time
        return CrawlResult(
            url=url,
            markdown="",
            success=False,
            error=str(e),
            metadata={"processing_time": processing_time},
        )


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

    progress = create_progress()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        with progress:
            # Track overall progress (indeterminate since we don't know total URLs)
            progress.add_task(f"[cyan]Recursive crawl from {start_url}", total=None)

            for depth in range(max_depth):
                if not current_urls:
                    break

                urls_to_crawl = [
                    url for url in current_urls if normalize_url(url) not in visited
                ]

                if not urls_to_crawl:
                    break

                # Create depth-specific task
                depth_task = progress.add_task(
                    f"[yellow]Depth {depth + 1}: Crawling {len(urls_to_crawl)} URLs",
                    total=len(urls_to_crawl),
                )

                async def crawl_with_progress(url):
                    result = await crawl_single_url(crawler, url, config)
                    progress.advance(depth_task)
                    return result

                # Crawl current level URLs
                crawl_tasks = [crawl_with_progress(url) for url in urls_to_crawl]
                batch_results = await asyncio.gather(
                    *crawl_tasks, return_exceptions=True
                )

                next_level_urls = set()

                for result in batch_results:
                    if isinstance(result, Exception):
                        error_panel = Panel(
                            f"[red]{str(result)}[/red]",
                            title="[red]❌ Crawl Error[/red]",
                            border_style="red",
                            padding=(0, 1),
                        )
                        console.print(error_panel)
                        continue

                    if not isinstance(result, CrawlResult):
                        continue

                    all_results.append(result)
                    visited.add(normalize_url(result.url))

                    if result.success and depth < max_depth - 1:
                        # Extract internal links for next level
                        try:
                            crawl_result = await crawler.arun(
                                url=result.url, config=config
                            )
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
                progress.remove_task(depth_task)

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

    progress = create_progress()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        with progress:
            # Process in batches to avoid overwhelming the system
            batch_size = max_concurrent
            all_results = []

            # Create overall progress task
            overall_task = progress.add_task(
                f"[cyan]Crawling {len(urls)} URLs from sitemap", total=len(urls)
            )

            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i : i + batch_size]
                batch_num = i // batch_size + 1

                # Create batch progress task
                batch_task = progress.add_task(
                    f"[yellow]Batch {batch_num}: Processing {len(batch_urls)} URLs",
                    total=len(batch_urls),
                )

                async def crawl_with_progress(url):
                    result = await crawl_single_url(crawler, url, config)
                    progress.advance(batch_task)
                    progress.advance(overall_task)
                    return result

                tasks = [crawl_with_progress(url) for url in batch_urls]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, CrawlResult):
                        all_results.append(result)
                    elif isinstance(result, Exception):
                        error_panel = Panel(
                            f"[red]{str(result)}[/red]",
                            title="[red]❌ Batch Processing Error[/red]",
                            border_style="red",
                            padding=(0, 1),
                        )
                        console.print(error_panel)

                # Remove completed batch task
                progress.remove_task(batch_task)

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
    successful_count = sum(map(is_successful, results))
    total_processing_time = sum(
        safe_get(r.metadata, "processing_time", 0.0) for r in results
    )

    return CrawlStats(
        total_urls=len(results),
        successful=successful_count,
        failed=len(results) - successful_count,
        total_chunks=sum(map(get_chunk_count, results)),
        processing_time=total_processing_time,
    )


def format_status(success: bool) -> str:
    """Format success status with color."""
    return "[green]✓[/green]" if success else "[red]✗[/red]"


def format_title(title: str) -> str:
    """Format title with fallback."""
    return title if title else "[dim]No title[/dim]"


def format_chunks_and_size(result: CrawlResult) -> Tuple[str, str]:
    """Format chunks and size display."""
    if result.success:
        chunk_count = get_chunk_count(result)
        char_count = get_char_count(result)
        return str(chunk_count), format_size(char_count)
    return "[red]0[/red]", "[red]0[/red]"


def format_time(processing_time: float) -> str:
    """Format processing time."""
    return f"{processing_time:.2f}s"


def format_preview(result: CrawlResult) -> str:
    """Format content preview."""
    if result.success and result.chunks:
        text = result.chunks[0].text
        preview = text[:80] + "..." if len(text) > 80 else text
        return f"[dim]{preview.replace(chr(10), ' ').replace(chr(13), ' ')}[/dim]"
    return "[red]No content[/red]" if not result.success else "[dim]Empty[/dim]"


def create_table_row(result: CrawlResult, show_content: bool) -> List[str]:
    """Create table row data for a result."""
    chunks_display, size_display = format_chunks_and_size(result)
    row = [
        format_status(result.success),
        result.url,
        format_title(result.title),
        chunks_display,
        size_display,
        format_time(safe_get(result.metadata, "processing_time", 0.0)),
    ]
    return row + ([format_preview(result)] if show_content else [])


def build_results_table(results: List[CrawlResult], show_content: bool) -> Table:
    """Build the results table."""
    table = Table(show_header=True, header_style="bold blue")
    columns = [
        ("Status", {"style": "center", "width": 8}),
        ("URL", {"style": "cyan", "overflow": "fold", "max_width": 50}),
        ("Title", {"style": "white", "max_width": 30, "overflow": "fold"}),
        ("Chunks", {"justify": "right", "style": "yellow", "width": 8}),
        ("Size", {"justify": "right", "style": "green", "width": 8}),
        ("Time", {"justify": "right", "style": "magenta", "width": 8}),
    ]

    if show_content:
        columns.append(
            ("Preview", {"style": "dim", "max_width": 40, "overflow": "fold"})
        )

    # Add columns
    for name, kwargs in columns:
        table.add_column(name, **kwargs)

    # Add rows
    for row_data in map(lambda r: create_table_row(r, show_content), results):
        table.add_row(*row_data)

    return table


def display_results(results: List[CrawlResult], show_content: bool = False) -> None:
    """Display crawl results with rich formatting."""
    stats = calculate_stats(results)
    avg_time = stats.processing_time / max(stats.total_urls, 1)

    # Summary panel
    summary = f"""[bold]URLs Processed:[/bold] {stats.total_urls}
[bold]Successful:[/bold] [green]{stats.successful}[/green]
[bold]Failed:[/bold] [red]{stats.failed}[/red]
[bold]Total Chunks:[/bold] {stats.total_chunks}
[bold]Processing Time:[/bold] {stats.processing_time:.2f}s ([dim]avg: {avg_time:.2f}s/URL[/dim])"""

    console.print(Panel(summary, title="📊 Crawl Summary", border_style="blue"))
    console.print(build_results_table(results, show_content))

    # Error details table
    failed_results, _ = partition(is_successful, results)
    failed_list = list(failed_results)

    if failed_list:
        console.print()
        error_table = Table(
            show_header=True, header_style="bold red", title="❌ Error Details"
        )
        error_table.add_column("URL", style="cyan", overflow="fold")
        error_table.add_column("Error", style="red", overflow="fold")

        for result in failed_list:
            error_table.add_row(result.url, result.error or "Unknown error")

        console.print(error_table)


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
        interrupt_panel = Panel(
            "[yellow]Crawling operation was interrupted by user[/yellow]",
            title="[yellow]⚠️  Interrupted[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
        console.print("\n")
        console.print(interrupt_panel)
        return 1
    except Exception as e:
        error_panel = Panel(
            f"[red]An unexpected error occurred:[/red]\n[red]{str(e)}[/red]",
            title="[red]❌ Fatal Error[/red]",
            border_style="red",
            padding=(0, 1),
        )
        console.print(error_panel)
        return 1


# Help display data structures
HELP_DATA = {
    "usage_examples": """# Basic single page crawling
uv run crawler.py https://example.com

# Recursive crawling with custom depth
uv run crawler.py https://site.com --recursive --depth 3

# High-speed sitemap processing
uv run crawler.py https://site.com/sitemap.xml --parallel 20

# Process markdown/text files
uv run crawler.py https://site.com/documentation.md

# Save results with custom filename
uv run crawler.py https://example.com --output my_results.xml

# Show content previews in results
uv run crawler.py https://example.com --show-content""",
    "strategies": [
        ("Single", "Extract content from one page", "Quick content extraction"),
        (
            "Recursive",
            "Follow internal links up to specified depth",
            "Site-wide content discovery",
        ),
        ("Sitemap", "Batch process URLs from XML sitemap", "Large-scale crawling"),
        (
            "Markdown",
            "Process text/markdown files directly",
            "Documentation processing",
        ),
    ],
    "options": [
        ("--recursive, -r", "Enable recursive crawling", "False"),
        ("--depth, -d", "Maximum crawl depth", "2"),
        ("--parallel, -p", "Concurrent requests", "5"),
        ("--chunk-size, -c", "Max chunk size (chars)", "1000"),
        ("--output, -o", "Output XML file path", "auto"),
        ("--quiet, -q", "Suppress detailed output", "False"),
        ("--show-content", "Show content previews", "False"),
    ],
    "tips": [
        "Use [cyan]--parallel 10-20[/cyan] for faster sitemap processing",
        "Set [cyan]--depth 1-2[/cyan] for recursive crawls to avoid deep recursion",
        "Use [cyan]--quiet[/cyan] when piping output to other tools",
        "Larger [cyan]--chunk-size[/cyan] values reduce chunk count but may impact processing",
        "XML output includes timing, metadata, and structured content chunks",
    ],
}


def create_table_from_data(
    data: List[Tuple[str, str, str]],
    headers: Tuple[str, str, str],
    styles: Tuple[str, str, str],
    header_style: str,
) -> Table:
    """Create a table from structured data."""
    table = Table(show_header=True, header_style=header_style)
    for header, style in zip(headers, styles):
        table.add_column(
            header, style=style, width=15 if header == headers[0] else None
        )

    for row in data:
        table.add_row(*row)
    return table


def display_help():
    """Display enhanced help screen with comprehensive information."""
    panels = [
        Panel(
            "[bold cyan]🕷️ Web Crawler Agent[/bold cyan]\n"
            "[dim]Extract and structure content from websites with multiple crawling strategies[/dim]",
            border_style="cyan",
            padding=(1, 2),
        ),
        Panel(
            Syntax(
                HELP_DATA["usage_examples"], "bash", theme="monokai", line_numbers=False
            ),
            title="[yellow]📝 Usage Examples[/yellow]",
            border_style="yellow",
            padding=(1, 1),
        ),
        Panel(
            create_table_from_data(
                HELP_DATA["strategies"],
                ("Strategy", "Description", "Best For"),
                ("cyan", "white", "green"),
                "bold blue",
            ),
            title="[blue]🎯 Crawling Strategies[/blue]",
            border_style="blue",
            padding=(1, 1),
        ),
        Panel(
            create_table_from_data(
                HELP_DATA["options"],
                ("Option", "Description", "Default"),
                ("cyan", "white", "yellow"),
                "bold green",
            ),
            title="[green]⚙️ Command Options[/green]",
            border_style="green",
            padding=(1, 1),
        ),
        Panel(
            "\n".join(f"• {tip}" for tip in HELP_DATA["tips"]),
            title="[magenta]💡 Performance Tips[/magenta]",
            border_style="magenta",
            padding=(1, 2),
        ),
    ]

    for i, panel in enumerate(panels):
        console.print(panel)
        if i < len(panels) - 1:
            console.print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        display_help()
        sys.exit(0)
    else:
        sys.exit(asyncio.run(main()))
