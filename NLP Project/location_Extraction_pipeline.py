import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification
import torch
from geopy.geocoders import Nominatim
import time

def extract_location_coordinates(tweet_df):
    # Load pre-trained BERT tokenizer and model for Named Entity Recognition
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

    # Create a list to store the extracted location coordinates
    extracted_coordinates = []

    # Initialize Nominatim geocoder
    geolocator = Nominatim(user_agent="geoapiExercises")

    for tweet in tweet_df["tweet_text"]:
        # Tokenize the tweet and convert tokens to IDs
        inputs = tokenizer.encode(tweet, return_tensors="pt")

        # Perform Named Entity Recognition (NER) using the BERT model
        with torch.no_grad():
            outputs = model(inputs)

        # Extract the predicted labels (entities) from the BERT model output
        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        # Find location coordinates based on predicted labels (we assume "B-LOC" indicates the start of location)
        coordinates = []
        for i, label_id in enumerate(predicted_labels):
            if label_id == 1:  # "B-LOC" label indicates the start of a location
                location_name = tokens[i].replace("#", "")  # Remove "#" symbol

                # Use Nominatim API to get location coordinates
                location = geolocator.geocode(location_name)
                if location:
                    coordinates.append((location.latitude, location.longitude))
                else:
                    coordinates.append((None, None))

                # Respect API usage policy (2 requests per second)
                time.sleep(0.5)

        # Add the list of coordinates to the extracted_coordinates list
        extracted_coordinates.append(coordinates)

    # Create a new DataFrame with the extracted coordinates
    extracted_df = pd.DataFrame({"location_coordinates": extracted_coordinates})

    return extracted_df

# Sample DataFrame containing tweets
data = {
    "tweet_text": [
        "Just arrived in New York City! #Travel",
        "Exploring the beautiful beaches of Maldives.",
        "Having a great time in Paris!",
        "Hiking in the mountains near Denver, Colorado.",
        "Missing Tokyo already. Can't wait to go back!",
    ]
}

tweet_df = pd.DataFrame(data)

# Call the function to extract location coordinates from the DataFrame
extracted_df = extract_location_coordinates(tweet_df)

# Print the extracted location coordinates
print(extracted_df)

# Save the extracted DataFrame to a new CSV file
extracted_df.to_csv("extracted_coordinates.csv", index=False)
