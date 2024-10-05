A Visual Question Answering (VQA) model is a system that answers questions about an image, using a combination of natural language processing and computer vision

Current model utilizes sklearn, cv2, torch, transformers, PIL, requests, and matplotlib

Behaviors:
- Color
- Object
- Shape

Why this is important:
To create more specified instances or classes of existing entities, the process would go as follows:
1. Use the given model to determine color, object, shape, etc
2. Use the given values to create a probability comparison of whether or not something in the image is what it is (i.e. I think this is a red cube, does the color and shape match?)
3. Add any number of specific objects relevent to the task at hand and assign them values of each trait, allowing for fine-tuned models and specific tasks

Future plans:
- Add at least 10 more predictors/behaviors
- Create secondary script for loading images of things, gathering its data, then cross comparing to the VQA model for future evaluations
- GUI
