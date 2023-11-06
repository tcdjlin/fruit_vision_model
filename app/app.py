
import torch
import gradio as gr
import os
from timeit import default_timer as timer
from typing import Tuple, Dict
from model import create_effnetb2_model

# Importing class names
with open('class_names.txt', 'r') as f:
    class_names = [fruit.strip() for fruit in f.readlines()]

# Initialising model
model, model_transforms = create_effnetb2_model(num_classes=len(class_names))

# Load model
model.load_state_dict(torch.load(f='fruits_100_feature_extractor_20.pth',
                                 map_location=torch.device('cpu')))

# Prediction
def predict(img) -> Tuple[Dict, float]:
    # Start a timer
    start_time = timer()
    
    # Transform the input image for use with EffNeetB2
    img = model_transforms(img).unsqueeze(0) # unsqueeze = add batch dimension on 0th index
    
    # Put model into eval mode, make prediction
    model.eval()
    with torch.inference_mode():
        # Pass transformed image through the model and turn the prediction logits into probabilities
        pred_probs = torch.softmax(model(img), dim=1)
        
    # Create a prediction label and prediction probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    
    # Return pred dict and pred time
    return pred_labels_and_probs, pred_time

# Launching demo
# Create title, description, article
title = 'Fruit Vision 👁️🔎🍊'
description = 'Trained 100 Fruit Classes with [EfficientNetB2](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2).'
article = 'Source code at [TC GitHub](https://github.com/andrewtclin/fruit_vision_model)'

# Create example list
example_list = [['examples/' + example] for example in os.listdir('examples')]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=5,
                                      label='Predictions'),
                             gr.Number(label='Prediction time (s)')],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo
demo.launch(debug=False, # print error locally
            share=False) # generate a publically shareable URL
