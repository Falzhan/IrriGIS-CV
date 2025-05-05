## Checks if images listed in the COCO JSON file are present in the directory


import os
import json
import shutil
from tkinter import Tk, filedialog
from pathlib import Path

def select_json_file():
    """Open a file dialog to select the COCO JSON file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select COCO instances_default.json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    return file_path

def move_unlisted_images(json_path):
    """Move images not listed in the JSON to a new folder"""
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all listed image file names
    listed_images = {img['file_name'].lower() for img in data['images']}  # Convert to lowercase
    print("\nImages listed in JSON:")
    for img in listed_images:
        print(f"- {img}")
    
    # Get directory of the JSON file
    json_dir = Path(json_path).parent
    
    # Create 'unlisted_images' directory if it doesn't exist
    output_dir = json_dir / 'unlisted_images'
    output_dir.mkdir(exist_ok=True)
    
    # Find all image files in the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_images = [
        f.name for f in json_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print("\nImages found in directory:")
    for img in all_images:
        print(f"- {img}")
    
    # Find and move unlisted images
    moved_count = 0
    for img_name in all_images:
        if img_name.lower() not in listed_images:  # Convert to lowercase for comparison
            src = json_dir / img_name
            dst = output_dir / img_name
            shutil.move(str(src), str(dst))
            moved_count += 1
            print(f"\nMoved: {img_name} (not found in JSON)")
        else:
            print(f"Keeping: {img_name} (found in JSON)")
    
    print(f"\nDone! Moved {moved_count} unlisted images to '{output_dir}'")

def main():
    print("Please select the COCO instances_default.json file")
    json_path = select_json_file()
    
    if not json_path:
        print("No file selected. Exiting.")
        return
    
    if not json_path.lower().endswith('.json'):
        print("Please select a JSON file. Exiting.")
        return
    
    try:
        move_unlisted_images(json_path)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please make sure you selected a valid COCO format JSON file.")

if __name__ == "__main__":
    main()