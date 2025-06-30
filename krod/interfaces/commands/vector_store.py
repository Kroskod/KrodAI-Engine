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

@vector_store.command()
@click.option('--collection', default='krod_documents', help='Collection name')
@click.option('--confirm', is_flag=True, help='Confirm recreation without prompting')
async def recreate(collection: str, confirm: bool):
    """Recreate a vector store collection (WARNING: deletes all data)"""
    if not confirm:
        click.confirm(f"This will DELETE ALL DATA in collection '{collection}'. Continue?", abort=True)
    
    try:
        vs = VectorStore({"collection_name": collection})
        await vs.recreate_collection()
        click.echo(f"Successfully recreated collection '{collection}'")
    except Exception as e:
        click.echo(f"Error recreating collection: {str(e)}", err=True)
        raise click.ClickException(str(e))

@vector_store.command()
@click.option('--file', required=True, type=click.Path(exists=True), help='File containing documents to ingest')
@click.option('--collection', default='krod_documents', help='Collection name')
@click.option('--batch-size', default=32, help='Batch size for ingestion')
async def ingest(file: str, collection: str, batch_size: int):
    """Ingest documents from a file into the vector store"""
    try:
        import json
        import csv
        from pathlib import Path
        
        vs = VectorStore({"collection_name": collection})
        file_path = Path(file)
        file_ext = file_path.suffix.lower()
        
        texts = []
        metadatas = []
        
        click.echo(f"Reading documents from {file}...")
        
        # Handle different file formats
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        texts.append(line)
                        metadatas.append({"source": file_path.name})
        
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            text = item.get('text', '') or item.get('content', '')
                            if text:
                                texts.append(text)
                                # Extract metadata excluding the text/content field
                                metadata = {k: v for k, v in item.items() 
                                           if k not in ['text', 'content']}
                                metadatas.append(metadata)
                        elif isinstance(item, str):
                            texts.append(item)
                            metadatas.append({"source": file_path.name})
        
        elif file_ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text_field = next((f for f in ['text', 'content', 'document'] 
                                     if f in row), None)
                    if text_field:
                        text = row[text_field]
                        metadata = {k: v for k, v in row.items() if k != text_field}
                        texts.append(text)
                        metadatas.append(metadata)
        
        else:
            raise click.ClickException(f"Unsupported file format: {file_ext}")
        
        if not texts:
            click.echo("No documents found in the file")
            return
        
        click.echo(f"Found {len(texts)} documents. Adding to collection '{collection}'...")
        
        # Add documents in batches
        total_docs = len(texts)
        with click.progressbar(length=total_docs, label='Ingesting documents') as bar:
            for i in range(0, total_docs, batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                await vs.add_documents(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                
                bar.update(len(batch_texts))
        
        click.echo(f"Successfully added {total_docs} documents to collection '{collection}'")
        
    except Exception as e:
        click.echo(f"Error ingesting documents: {str(e)}", err=True)
        raise click.ClickException(str(e))
