# Sentiment Analysis System LAB-P1 - 9
## About the project
### Objective
The goal of this project is to design and implement a sentiment analysis system based
on a predefined sentiment dictionary. Given a text document containing multiple
paragraphs and sentences, the system should process the text and compute sentiment
scores. Students will practice decomposing a real-world problem into programming logic
and algorithms, collaborate using GitHub, and produce a final system with both backend
logic and a simple frontend visualization.

### Dataset
For a sentiment analysis system, you need a dataset of sufficient size to test and
demonstrate its effectiveness. If you choose to use a machine learning approach, you
must have a training set (to build the model) and a test set (to evaluate it). The training
set and test set must not overlap to ensure a fair and reliable evaluation.

### Requirements
1. Calculate the sentiment score of each sentence using the provided dictionary.
2. Identify the most positive and most negative sentences in the entire text.
3. Apply a sliding window over paragraphs (e.g., 3 sentences per window) to
determine the most positive and most negative paragraph segments.
Follow-up: If you also want to know which exact sentences form these segments,
how would you modify the algorithm?
4. Without fixing the window size, find the most positive and most negative
continuous paragraph segments of arbitrary length.
Follow-up: If you also want to know which exact sentences form these segments,
how would you modify the algorithm?
5. Suppose you have an English dictionary. During data processing, all spaces in a
sentence were accidentally removed (e.g., "thisisapen"). Re-insert spaces to find
a possible valid segmentation.
Follow-up: If multiple valid segmentations exist, how would you return all possible
combinations?

## Overview
### Running the project
### How it works

We adopted a three-class sentiment formulation (negative, neutral, positive) to preserve the original rating semantics and to better support sliding-window sentiment analysis, where neutral sentences play an important role in modelling contextual transitions.

### License
MIT — free to use and adapt with attribution.
45 47 52 52 53 55 56 58 62

python -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process
.venv\Scripts\activate
python -m pip install -U pip

pip install pandas scikit-learn datasets transformers evaluate fastapi uvicorn "uvicorn[standard]"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uvicorn app:app --reload
