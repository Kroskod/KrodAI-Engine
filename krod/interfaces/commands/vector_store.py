import click
# from typing import Optional
from krod.core.vector_store import VectorStore



@click.group()
def vector_store():
    """
    manage vector store operations for krod
    """
    pass


@vector_store.command()
@click.option('--text', prompt='Document text', help='Text to add to the vector store')
@click.option('--collection', default='krod_documents', help='Collection name')
def add(text: str, collection: str):
    """Add a document to the vector store"""
    try:
        vs = VectorStore({"collection_name": collection})
        doc_id = vs.add_document(text)
        click.echo(f"Added document with ID: {doc_id}")
    except Exception as e:
        click.echo(f"Error adding document: {str(e)}", err=True)
        raise click.ClickException(str(e))

@vector_store.command()
@click.option('--query', prompt='Search query', help='Text to search for')
@click.option('--top-k', default=3, help='Number of results to return')
@click.option('--collection', default='krod_documents', help='Collection name')
def search(query: str, top_k: int, collection: str):
    """Search the vector store"""
    try:
        vs = VectorStore({"collection_name": collection})
        results = vs.search(query, top_k=top_k)
    except Exception as e:
        click.echo(f"Error searching vector store: {str(e)}", err=True)
        raise click.ClickException(str(e))

    if not results:
        click.echo("No results found")
        return

    click.echo(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (Score: {result['similarity']:.3f}) ---")
        click.echo(f"ID: {result['id']}")
        click.echo(f"Text: {result['text'][:200]}..." if len(result['text']) > 200 else result['text'])
        if result.get('metadata'):
            click.echo("Metadata:")
            for k, v in result['metadata'].items():
                click.echo(f"  {k}: {v}")

@vector_store.command()
@click.option('--collection', default='krod_documents', help='Collection name')
def count(collection: str):
    """Count documents in the vector store"""
    try:
        vs = VectorStore({"collection_name": collection})
        count = vs.count_documents()
        click.echo(f"Found {count} documents in collection '{collection}'")
    except Exception as e:
        click.echo(f"Error counting documents: {str(e)}", err=True)
        raise click.ClickException(str(e))


