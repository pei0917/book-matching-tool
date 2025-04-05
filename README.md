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
2. Install PyTorch manually (based on your CUDA version).
   - Important: If you have CUDA 12.6 or newer, install a PyTorch version that supports up to CUDA 12.4 (as some packages might not be compatible with 12.6+).
3. Download the YOLOv11 model (yolo11n-seg.pt) manually and place it in the project folder.
4. Run the inference.py script to detect and match books:
    ```bash
    python inference.py
    ```
5. The output image with detected books and their matches will be saved as output_1.jpg in the examples/ folder.

## Notes
- The YOLOv11 model (yolo11n-seg.pt) is used for detecting books in the input image.
- CLIP is used to match detected books with reference images provided in the reference_images dictionary in inference.py.
- Example images for testing are located in the examples/ folder.
