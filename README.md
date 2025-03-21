# AI-Trip-planner
This project allows users to generate detailed travel itineraries based on either a number of days for a trip or a date range. The app integrates with OpenAI’s GPT-3.5 Turbo model to create personalized travel plans, including suggestions for tourist spots, activities, restaurants, and local holiday events.

## Features

- Generate a trip itinerary based on a number of days.
- Generate a trip itinerary for a specific date range.
- Local holiday events are included in the itinerary if applicable.
- Gradio interface for easy interaction.

## Requirements

- Python 3.x
- OpenAI API Key
- Gradio
- `requests`
- `openai`
- `re`
- `datetime`
  
You can install the necessary dependencies using the following command:

pip install openai gradio

## Usage
Once the app is running, enter a prompt like the following to generate your travel itinerary:

For a number of days trip: Example: plan 5-day trip to Paris

For a date range trip: Example: plan Paris trip from April 1 to April 5

The app will respond with a detailed itinerary including recommendations for tourist spots, activities, and local events happening during the trip.
