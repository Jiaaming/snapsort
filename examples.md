# SnapsSort CLI Usage Examples

## Semantic Search

Find images based on a text description:

```bash
# Search for images containing animals
snapsort search animals
# Search for images containing cars and people
snapsort search "car and people"
# Search for landscape images (limit to top 5 results)
snapsort search landscape -n 5
```

## Find Similar Image

Find images similar to a reference image:

```bash
# Find images similar to a specific image
snapsort find /path/to/reference/image.jpg
```

## Exposure Classification

Sort images by exposure levels:

```bash
# Retrieve images with different exposure characteristics
snapsort tone --level low      # Dark or low-key images 
snapsort tone --level mid      # Well-exposed or balanced images
snapsort tone --level high     # Bright or high-key images
```

## Redo last command
```bash
# Revert the folder structure to the state before the last command
snapsort redo
```
## Key Features

- **Semantic Search**: Discover images using natural language descriptions
- **Similar Image Finder**: Locate visually similar images
- **Exposure Sorting**: Classify images by their light and tone characteristics

**Note**: Ensure you're in the target directory before running these commands.
