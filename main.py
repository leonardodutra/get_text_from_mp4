import cv2
import easyocr
import numpy as np

# Initialize EasyOCR reader (specify languages, e.g., ['en'] for English)
reader = easyocr.Reader(['en'])

def recognize_plate(frame):
    """
    In a real application, you'd first use an object detection model
    (e.g., YOLO) to find and crop the plate region before this step.
    """
    # For this simple example, we assume the plate is in the frame
    # For better results, apply image processing (grayscale, contrast, etc.)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR
    results = reader.readtext(gray_frame)
    
    plate_text = ""
    for (bbox, text, prob) in results:
        # Filter for probable license plate formats and high confidence
        # (this part requires fine-tuning for specific plate formats)
        if prob > 0.5 and len(text) > 3:
            plate_text += text + " "
    
    return plate_text.strip()

def process_video(video_path, output_log="license_plates.txt"):
    """
    Reads an MP4 file frame by frame and attempts to recognize license plates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print(f"Processing video: {video_path}...")
    detected_plates = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video
        
        # Process frame (e.g., every 10th frame to save time/resources)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
            plate_text = recognize_plate(frame)
            if plate_text and plate_text not in detected_plates:
                detected_plates.add(plate_text)
                print(f"Detected Plate: {plate_text}")
    
    cap.release()
    
    # Save results to a file
    with open(output_log, "w") as f:
        for plate in sorted(list(detected_plates)):
            f.write(f"{plate}\n")
    print(f"Processing complete. Detected plates saved to {output_log}")

# Example Usage:
# Replace "your_video.mp4" with the path to your MP4 file
process_video("your_video.mp4")
