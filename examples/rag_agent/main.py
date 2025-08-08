#!/usr/bin/env python3
"""
PydanticAI RAG Agent CLI Interface.

This module provides a command-line interface for interacting with
the RAG agent, including document management and querying.
"""

import asyncio
import os
from typing import Optional, List
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from src.agents.rag_agent import PydanticRAGAgent, RAGQuery, RAGContext
from src.models.document import Document, DocumentType, DocumentMetadata
from config.settings import get_settings

app = typer.Typer(help="PydanticAI RAG Agent CLI")
console = Console()


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        "logs/rag_agent.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        lambda msg: None,  # Suppress console output in CLI mode
        level="ERROR"
    )


async def create_agent() -> PydanticRAGAgent:
    """Create and initialize RAG agent."""
    settings = get_settings()
    
    # Validate settings
    issues = settings.validate_settings()
    if issues:
        console.print("[red]Configuration issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        raise typer.Exit(1)
    
    agent = PydanticRAGAgent(settings)
    return agent


def load_document_from_file(file_path: str) -> Document:
    """Load document from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine document type
    suffix = path.suffix.lower()
    doc_type_map = {
        ".txt": DocumentType.TEXT,
        ".md": DocumentType.MARKDOWN,
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".html": DocumentType.HTML
    }
    
    doc_type = doc_type_map.get(suffix, DocumentType.TEXT)
    
    # Read content (simplified - in practice, you'd use proper parsers)
    if doc_type == DocumentType.PDF:
        # Would use pypdf or similar
        content = f"[PDF content from {file_path}]"
    elif doc_type == DocumentType.DOCX:
        # Would use python-docx
        content = f"[DOCX content from {file_path}]"
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
    metadata = DocumentMetadata(
        title=path.stem,
        file_size=path.stat().st_size,
        tags=[doc_type.value]
    )
    
    return Document(
        content=content,
        document_type=doc_type,
        source_path=str(path.absolute()),
        metadata=metadata
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the RAG agent"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
):
    """Query the RAG agent with a question."""
    setup_logging()
    
    async def run_query():
        try:
            agent = await create_agent()
            
            # Switch provider if specified
            if provider:
                success = await agent.switch_provider(provider)
                if not success:
                    console.print(f"[red]Failed to switch to provider: {provider}[/red]")
                    return
                console.print(f"[green]Switched to provider: {provider}[/green]")
            
            # Prepare query
            rag_context = RAGContext(metadata={"cli": True})
            if context:
                rag_context.metadata["additional_context"] = context
            
            rag_query = RAGQuery(question=question, context=rag_context)
            
            if stream:
                # Stream response
                console.print(f"[bold blue]Question:[/bold blue] {question}")
                console.print(f"[bold green]Answer:[/bold green]", end=" ")
                
                async for chunk in agent.stream_query(rag_query):
                    console.print(chunk, end="")
                console.print()  # New line at end
                
            else:
                # Regular response
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing query...", total=None)
                    response = await agent.query(rag_query)
                    progress.remove_task(task)
                
                # Display response
                console.print(Panel(
                    f"[bold blue]Question:[/bold blue] {question}\n\n"
                    f"[bold green]Answer:[/bold green] {response.answer}",
                    title="RAG Response",
                    border_style="blue"
                ))
                
                # Display sources if available
                if response.sources:
                    table = Table(title="Sources", show_header=True, header_style="bold magenta")
                    table.add_column("Document", style="cyan")
                    table.add_column("Chunk", style="green")
                    table.add_column("Similarity", style="yellow")
                    table.add_column("Content Preview", style="white")
                    
                    for source in response.sources[:5]:  # Show top 5 sources
                        content_preview = source.content[:100] + "..." if len(source.content) > 100 else source.content
                        table.add_row(
                            source.document_metadata.title or "Unknown",
                            str(source.chunk_metadata.chunk_index),
                            f"{source.similarity_score:.3f}" if source.similarity_score else "N/A",
                            content_preview
                        )
                    
                    console.print(table)
                
                # Display metadata
                console.print(f"[dim]Confidence: {response.confidence:.3f} | "
                            f"Provider: {response.metadata.get('model_used', 'Unknown')} | "
                            f"Sources: {len(response.sources)}[/dim]")
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_query())


@app.command()
def add_document(
    file_path: str = typer.Argument(..., help="Path to document file"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Document title"),
    author: Optional[str] = typer.Option(None, "--author", "-a", help="Document author"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
):
    """Add a document to the knowledge base."""
    setup_logging()
    
    async def run_add():
        try:
            agent = await create_agent()
            
            # Load document
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading document...", total=None)
                document = load_document_from_file(file_path)
                progress.remove_task(task)
            
            # Update metadata if provided
            if title:
                document.metadata.title = title
            if author:
                document.metadata.author = author
            if tags:
                document.metadata.tags = [tag.strip() for tag in tags.split(",")]
            
            # Add to knowledge base
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Adding to knowledge base...", total=None)
                result = await agent.add_documents([document])
                progress.remove_task(task)
            
            if "error" in result:
                console.print(f"[red]Error adding document: {result['error']}[/red]")
            else:
                console.print(f"[green]Successfully added document:[/green]")
                console.print(f"  • Document ID: {document.id}")
                console.print(f"  • Title: {document.metadata.title}")
                console.print(f"  • Chunks created: {result.get('total_chunks', 0)}")
                console.print(f"  • Word count: {document.get_word_count()}")
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_add())


@app.command()
def add_directory(
    directory_path: str = typer.Argument(..., help="Path to directory containing documents"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process subdirectories"),
    file_types: Optional[str] = typer.Option("txt,md,pdf", "--types", help="File types to process"),
):
    """Add all documents from a directory to the knowledge base."""
    setup_logging()
    
    async def run_add_directory():
        try:
            agent = await create_agent()
            
            directory = Path(directory_path)
            if not directory.exists():
                console.print(f"[red]Directory not found: {directory_path}[/red]")
                raise typer.Exit(1)
            
            # Get file extensions to process
            extensions = [f".{ext.strip()}" for ext in file_types.split(",")]
            
            # Find files
            files = []
            if recursive:
                for ext in extensions:
                    files.extend(directory.rglob(f"*{ext}"))
            else:
                for ext in extensions:
                    files.extend(directory.glob(f"*{ext}"))
            
            if not files:
                console.print(f"[yellow]No files found with extensions: {extensions}[/yellow]")
                return
            
            console.print(f"[blue]Found {len(files)} files to process[/blue]")
            
            # Process files
            documents = []
            failed_files = []
            
            with Progress(console=console) as progress:
                task = progress.add_task("Loading documents...", total=len(files))
                
                for file_path in files:
                    try:
                        document = load_document_from_file(str(file_path))
                        documents.append(document)
                        progress.console.print(f"  ✓ Loaded: {file_path.name}")
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
                        progress.console.print(f"  ✗ Failed: {file_path.name} - {e}")
                    
                    progress.advance(task)
            
            if documents:
                # Add to knowledge base
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Adding to knowledge base...", total=None)
                    result = await agent.add_documents(documents)
                    progress.remove_task(task)
                
                console.print(f"[green]Successfully processed directory:[/green]")
                console.print(f"  • Documents added: {result.get('added_documents', 0)}")
                console.print(f"  • Total chunks: {result.get('total_chunks', 0)}")
                console.print(f"  • Failed documents: {len(result.get('failed_documents', []))}")
                
                if failed_files:
                    console.print(f"[yellow]Failed to load {len(failed_files)} files:[/yellow]")
                    for file_path, error in failed_files:
                        console.print(f"  • {file_path}: {error}")
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_add_directory())


@app.command()
def stats():
    """Show RAG agent statistics."""
    setup_logging()
    
    async def run_stats():
        try:
            agent = await create_agent()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Gathering statistics...", total=None)
                stats = await agent.get_stats()
                progress.remove_task(task)
            
            # Display statistics
            table = Table(title="RAG Agent Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Agent Name", stats.get("agent_name", "Unknown"))
            table.add_row("LLM Provider", stats.get("provider", "Unknown"))
            
            vector_stats = stats.get("vector_store", {})
            table.add_row("Total Documents", str(vector_stats.get("total_documents", 0)))
            table.add_row("Total Chunks", str(vector_stats.get("total_chunks", 0)))
            table.add_row("Embedding Model", str(vector_stats.get("embedding_model", "Unknown")))
            
            settings_info = stats.get("settings", {})
            table.add_row("Top K", str(settings_info.get("top_k", "Unknown")))
            table.add_row("Similarity Threshold", str(settings_info.get("similarity_threshold", "Unknown")))
            table.add_row("Chunk Size", str(settings_info.get("chunk_size", "Unknown")))
            
            console.print(table)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_stats())


@app.command()
def health():
    """Check RAG agent health."""
    setup_logging()
    
    async def run_health():
        try:
            agent = await create_agent()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Checking health...", total=None)
                health_status = await agent.health_check()
                progress.remove_task(task)
            
            # Display health status
            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "unhealthy": "red"
            }.get(health_status["status"], "white")
            
            console.print(Panel(
                f"[bold {status_color}]Status: {health_status['status'].upper()}[/bold {status_color}]\n"
                f"Timestamp: {health_status['timestamp']}",
                title="Health Check",
                border_style=status_color
            ))
            
            # Component status
            if "components" in health_status:
                table = Table(title="Component Status", show_header=True, header_style="bold magenta")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="white")
                table.add_column("Details", style="dim")
                
                for component, info in health_status["components"].items():
                    status = info["status"]
                    status_style = {
                        "healthy": "[green]✓ Healthy[/green]",
                        "unhealthy": "[red]✗ Unhealthy[/red]",
                        "degraded": "[yellow]⚠ Degraded[/yellow]"
                    }.get(status, status)
                    
                    details = []
                    if "provider" in info:
                        details.append(f"Provider: {info['provider']}")
                    if "document_count" in info:
                        details.append(f"Documents: {info['document_count']}")
                    if "error" in info:
                        details.append(f"Error: {info['error']}")
                    
                    table.add_row(
                        component.replace("_", " ").title(),
                        status_style,
                        " | ".join(details)
                    )
                
                console.print(table)
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_health())


@app.command()
def providers():
    """List available LLM providers."""
    setup_logging()
    
    settings = get_settings()
    providers = settings.get_llm_providers()
    
    if not providers:
        console.print("[yellow]No LLM providers configured[/yellow]")
        return
    
    table = Table(title="Available LLM Providers", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Status", style="white")
    table.add_column("Temperature", style="yellow")
    table.add_column("Max Tokens", style="blue")
    
    for name, config in providers.items():
        status = "[green]✓ Enabled[/green]" if config.enabled else "[red]✗ Disabled[/red]"
        if name == settings.default_provider:
            status += " [bold](Default)[/bold]"
        
        table.add_row(
            name,
            config.model,
            status,
            str(config.temperature),
            str(config.max_tokens)
        )
    
    console.print(table)


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    app()

