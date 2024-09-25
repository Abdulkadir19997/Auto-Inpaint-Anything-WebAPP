import os

def print_directory_tree(startpath, indent="", exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ["inpaint_anything", "__pycache__", "results"]

    # Iterate through all items in the given startpath
    for item in os.listdir(startpath):
        item_path = os.path.join(startpath, item)
        
        # Skip excluded directories
        if item in exclude_dirs:
            continue
        
        # Print the current item with proper indentation
        print(indent + "├── " + item)
        
        if os.path.isdir(item_path):
            # Recursively print sub-directories with increased indentation
            print_directory_tree(item_path, indent + "│   ", exclude_dirs)

# Path to your project folder (replace with your project root path if needed)
project_folder = "."
print_directory_tree(project_folder)
