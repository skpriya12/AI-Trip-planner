
import openai
import re
from datetime import datetime
import gradio as gr
import os
# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_input(input_string):
    # Regular expression to match both "X-day trip" and "from date to date" formats
    trip_pattern = r'plan (\d+)-day trip to (\w+)'  # Matches "plan 5-day trip to Paris"
    date_pattern = r'plan (\w+) trip from (\w+ \d{1,2}) to (\w+ \d{1,2})'  # Matches "plan Paris trip from April 1 to April 2"

    # Check for "X-day trip" format
    match_trip = re.match(trip_pattern, input_string)
    if match_trip:
        days = int(match_trip.group(1))
        destination = match_trip.group(2)
        return "trip_days", days, destination

    # Check for "from date to date" format
    match_dates = re.match(date_pattern, input_string)
    if match_dates:
        destination = match_dates.group(1)
        start_date = datetime.strptime(match_dates.group(2), "%B %d")
        end_date = datetime.strptime(match_dates.group(3), "%B %d")
        return "trip_dates", destination, start_date, end_date

    return None, None, None


def generate_itinerary(destination, days, start_date=None, end_date=None):
    prompt = f"Create a detailed {days}-day itinerary for a trip to {destination}. Include popular tourist spots, activities, recommended restaurants for both lunch and dinner, and any local holiday events that may be happening during the trip."

    if start_date and end_date:
        prompt += f" The trip is from {start_date.strftime('%B %d')} to {end_date.strftime('%B %d')}."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Now using GPT-3.5 Turbo
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,  # Adjust token count as needed
            temperature=0.7  # Set the creativity level (0.0-1.0)
        )

        itinerary = response['choices'][0]['message']['content'].strip()
        return itinerary
    except Exception as e:
        return f"Error generating itinerary: {str(e)}"


def generate_trip(input_string):
    trip_type, *params = parse_input(input_string)

    if trip_type == "trip_days":
        days, destination = params
        return generate_itinerary(destination, days)
    elif trip_type == "trip_dates":
        destination, start_date, end_date = params
        return generate_itinerary(destination, (end_date - start_date).days + 1, start_date, end_date)
    else:
        return "Sorry, I couldn't understand your request."


# Gradio Interface
def gradio_interface(input_string):
    return generate_trip(input_string)


# Launch Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="Travel Itinerary Generator",
    description="Enter a prompt like 'plan 5-day trip to Paris' or 'plan Paris trip from April 1 to April 5' to get a detailed itinerary. The itinerary includes local holiday events if applicable."
)

iface.launch()
