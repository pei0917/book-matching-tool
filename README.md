# Book Matching Tool

This project provides a tool to detect and match books in an image using YOLOv11 for object detection and CLIP for feature matching.

## Folder Structure

- **inference.py**: Script to run the book matching process.
- **v1.py**: Contains the core logic for book detection, perspective transformation, and matching.
- **requirements.txt**: Lists the dependencies required for the project.
- **examples/**: Contains example images for testing the tool.

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the YOLOv8 model (yolo11n-seg.pt) manually and place it in the project folder.
3. Run the inference.py script to detect and match books:
    ```bash
    python inference.py
    ```
4. The output image with detected books and their matches will be saved as output_1.jpg in the examples/ folder.

## Notes
- The YOLOv8 model (yolo11n-seg.pt) is used for detecting books in the input image.
- CLIP is used to match detected books with reference images provided in the reference_images dictionary in inference.py.
- Example images for testing are located in the examples/ folder.