from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import cv2

def detect_objects(image_path):
    # loading the pretrained model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # open image
    image = Image.open(image_path)

    # preprocess and run through the model
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # extract predictions
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # display results
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.5:  # Filter out objects with low confidence
            box = [round(i, 2) for i in box.tolist()]
            detected_objects.append((model.config.id2label[label.item()], score.item(), box))

    return detected_objects

def display_image_with_boxes(image_path, objects):
    image = cv2.imread(image_path)
    for obj, score, box in objects:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{obj}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
