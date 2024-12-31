# AI-Powered Image Captioning Bot

## Overview
This project is an AI-powered image captioning bot that generates descriptive captions for images using the MS COCO dataset and CLIP-GPT architecture.

## Features
- Leverages OpenAI's CLIP model for image encoding.
- Uses GPT-2 for natural language generation.
- Includes a user-friendly web interface for uploading images and viewing captions.

## Directory Structure
AI-Image-Captioning-Bot/

├── app/

│   ├── __init__.py

│   ├── api.py

│   ├── inference.py

├── models/

│   ├── clip_gpt_bridge.py

│   ├── model.py

├── utils/

│   ├── dataset.py

│   ├── preprocess.py

├── static/

│   ├── index.html

├── uploads/

├── requirements.txt

├── README.md

├── train.py

└── .gitignore


## Setup
1. Clone this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the Flask app:

    `python -m flask run`

Access the interface at http://127.0.0.1:5000