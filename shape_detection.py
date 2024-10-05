import cv2

def detect_shape(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate contour shape
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        shape = "unknown"
        
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"  # This can also be a rectangle depending on aspect ratio
        elif len(approx) > 4:
            shape = "circle"
        
        return shape
    
    return "No shape detected"
