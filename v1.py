import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from ultralytics import YOLO
from sklearn.neighbors import NearestNeighbors

def detect_books(image_path):
    """Detect books using YOLOv8."""
    yolo_model = YOLO("yolo11n-seg.pt")
    results = yolo_model(image_path, verbose=False)
    detected_books = []
    
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 73:  # COCO dataset class ID for "book"
                x1, y1, x2, y2 = map(int, box.tolist())
                detected_books.append((x1, y1, x2, y2))
    
    return detected_books

def get_corners(binary_img):
    """Find corners of the book using contours and approxPolyDP."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None

def perspective_transform(image):
    """Warp the book to make it rectangular."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    corners = get_corners(binary)
    if corners is None:
        return image  # Return original if corners not found

    # Order points: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    # Define new rectangle dimensions
    width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
    
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    
    return cv2.warpPerspective(image, matrix, (int(width), int(height)))

def match_books(image_path, reference_images, output_path="output_1.jpg"):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model.to(device)

    # Load reference book images
    reference_features = {}
    image = cv2.imread(image_path)
    detected_books = detect_books(image_path)
    matched_books = {}
    output = ""
    # Save and dispaly the final image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)  # Green color for bounding boxes
    detected_img = cv2.imread(image_path)

    # Extract CLIP features for reference books
    for name, path in reference_images.items():
        img = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            reference_features[name] = clip_model.encode_image(img).cpu().numpy()


    for i, (x1, y1, x2, y2) in enumerate(detected_books):
        book_crop = image[y1:y2, x1:x2]
        calibrated_book = perspective_transform(book_crop)  # Rectify the book shape
        
        # Deep Learning CLIP Matching
        book_pil = Image.fromarray(cv2.cvtColor(calibrated_book, cv2.COLOR_BGR2RGB))
        book_clip = preprocess(book_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            book_embedding = clip_model.encode_image(book_clip).cpu().numpy()

        reference_features_array = np.array(list(reference_features.values()))
        reference_features_array = reference_features_array.squeeze(axis=1)
        knn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(list(reference_features_array))
        book_embedding = book_embedding.squeeze()
        _, indices = knn.kneighbors([book_embedding])
        best_clip_match = list(reference_features.keys())[indices[0][0]]
        
        # Store results
        matched_books[i] = {
            "bbox": (x1, y1, x2, y2),
            "clip_match": best_clip_match,
        }
        
        output += best_clip_match + " "

        # Draw the bounding box
        cv2.rectangle(detected_img, (x1, y1), (x2, y2), color, thickness)

        # Put the detected file name as text
        text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
        cv2.putText(detected_img, detected_filename, text_position, font, font_scale, color, thickness)

    cv2.imwrite(output_path, detected_img)

    return output