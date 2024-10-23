from color_detection import detect_dominant_color
from object_detection import detect_objects
from shape_detection import detect_shape

def answer_question(image_path, question_type):
    if "color" in question_type:
        color = detect_dominant_color(image_path)
        return f"The dominant color is {color}"
    elif "object" in question_type:
        objects = detect_objects(image_path)
        if objects:
            return "\n".join([f"Object: {obj}, Confidence: {conf * 100:.2f}%, Box: {box}" for obj, conf, box in objects])
        else:
            return "No objects detected."
    elif "shape" in question_type:
        shape = detect_shape(image_path)
        return f"The detected shape is {shape}"
    else:
        return "I don't understand that question. Please ask about color, object, or shape for now."

if __name__ == "__main__":

    # take input for image path
    image_path = input("Enter the image path: ")

    # take input for question type
    print("What do you want to ask? Options: [color, object, shape]")
    question = input("Enter your question: ").lower()

    result = answer_question(image_path, question)
    print(f"Answer: {result}")
