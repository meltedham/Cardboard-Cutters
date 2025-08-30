# Team name: Cardboard Cutters
For Tiktok TechJam 2025

================================================================================
# Project overview

Problem statement:
Filtering the Noise: ML for Trustworthy Location Reviews

Libraries used:
pandas
transformers
joblib
sklearn
torch
datasets


AI training protocol:

    1. A script is run on the dataset, accomplishing three functions: removing reviews containing links, standardising the whitespace characters by removing extra spaces, removing non-printable characters, and finally, removing duplicate . The result is that the reviews can now be compared and processed as strings.

    2. Labels were manually assigned to each review, sorting them into different classifications based on the language characteristics and contents of the review, in order to determine the effectiveness of each review. Reviews that are suspicious or unproductive are labelled accordingly.
    - Labels: Valid review, Irrelevant, Advertisement, Rant without visit

    3. The labelled data is used to train our model to associate reviews to one of the labels. This allows it to recognise and identify legitimate reviews and ignore reviews that are irrelevant, unhelpful, or likely AI-generated or spam.

================================================================================

# Setup instructions

1. Install the necessary libraries 
2. Open dashboard.py and run "python -m streamlit run dashboard.py"
3. Upload dataset to programme through user interface
4. The programme will process the uploaded dataset and output a file with only the reviews identified as valid reviews

================================================================================

# How to reproduce results

Follow the setup instructions, uploading the attached dataset "demonstration.csv" to the programme

================================================================================

# Team member contributions

- Manual labelling of about 1055 reviews split amongst each member

Refer to Github commit history at: https://github.com/meltedham/Cardboard-Cutters.git

================================================================================

# Devpost Submission


## Issue

Our solution addresses the issue of irrelevant reviews on Google that do not provide effective feedback on restaurants/locations. As a reader intending to use Google reviews to learn more about a place, I would want to filter out any pointless reviews so that I can focus on those that matter.

## How our solution helps

Our ML model filters out irrelevant reviews based on a few potential violations - irrelevance, advertisements, or rants without visits. These were selected based on the examples provided on Tiktok's Devpost problem statement. We discussed and agreed that these issues were the main culprits of bad reviews.

Our program first takes in a csv file of reviews, and cleans up all reviews. 

Cleaning:
- URLs removed
- Whitespaces standardised
- Non-printable characters removed
- Reviews without text removed
- Reviews less than 3 words removed
- Duplicate reviews removed

This effectively detects irrelevant content and labels them as such, filtering out the "noisy" reviews.

We mainly used Python as our main language for writing the program and training the model. VS Code was our editor of choice as it is modular and easy to set up. We also used GitHub for version control and code review.

We largely used Hugging Face transformers to import models such as roberta-base-openai-detector and bert-base-uncased for training, Scikit-learn for training data, and pandas for preprocessing.

As for datasets, we used Google Local Reviews datasets and labelled them manually with tags for model training.