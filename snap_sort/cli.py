# classify.py
import click
import logging
from snap_sort.exposure import classify_overexposed_images
from snap_sort.find_similar import find_similar_images
from snap_sort.redo import redo_last_operation
from snap_sort.semantic import semantic_search_images
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
@click.group(invoke_without_command=True)
@click.pass_context
def snapsort(ctx):
    """SnapSort command-line tool for image classification and organization."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help()) 
@snapsort.command(name='oe', short_help='Detect and move overexposed images')
@click.argument('folder_path', default='.')
def overexposed(folder_path):
    """Classify images in the specified FOLDER_PATH."""
    logging.info(f"Detecting overexposed images in: {folder_path}")
    classify_overexposed_images(folder_path)

@snapsort.command(name='similar', short_help='Find top N most similar images')
@click.option('--top-n', '-n', default=10, help='Number of most similar images to select')
@click.argument('photo_path')
@click.argument('folder_path', default='.')
def similar(top_n, photo_path, folder_path):
    """Find top N most similar images in FOLDER_PATH to PHOTO_PATH."""
    nums = find_similar_images(photo_path, folder_path, top_n)
    logging.info(f"Found {nums} similar images")

@snapsort.command(name='redo', short_help='Redo the last operation')
def redo():
    redo_last_operation()

@snapsort.command(name='search', short_help='Semantic search for images')
@click.option('--top-n', '-n', default=10, help='Number of most similar images to select')
@click.argument('prompt')
@click.argument('folder_path', default='.')
def similar(top_n, prompt, folder_path):
    semantic_search_images(prompt, folder_path, top_n)


if __name__ == "__main__":
    snapsort()
