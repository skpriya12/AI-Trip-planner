# ===============================================
# STEP 1: Imports & Setup
# ===============================================
import os, json, re, warnings
import gradio as gr
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load API keys from Hugging Face Spaces Secrets (Settings → Repository secrets)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, conlist
from typing import List, Optional

# ===============================================
# STEP 2: Schemas
# ===============================================
class Activity(BaseModel):
    name: str
    location: str
    description: str
    date: str
    cuisine: Optional[str] = None
    why_its_suitable: str
    reviews: Optional[List[str]] = None
    rating: Optional[float] = None

class AirlineOption(BaseModel):
    airline: str
    flight_number: str
    departure: str
    arrival: str
    price: str

class DayPlan(BaseModel):
    date: str
    activities: conlist(Activity, min_length=1)
    restaurants: conlist(str, min_length=1)
    flight: Optional[List[AirlineOption]] = None

class Itinerary(BaseModel):
    name: str
    day_plans: List[DayPlan]
    hotel: str

# ===============================================
# STEP 3: Agent
# ===============================================
groq_model = "groq/gemma2-9b-it"

itinerary_compiler = Agent(
    role="Itinerary Compiler",
    goal="Generate real activities, restaurants, and airline options for trips.",
    backstory="You are a professional travel planner who outputs structured itineraries with flight recommendations.",
    tools=[],
    llm=groq_model,
    verbose=True,
    allow_delegation=False,
)

# ===============================================
# STEP 4: Helpers
# ===============================================
def ensure_dict(obj):
    """Ensure CrewAI output is parsed JSON."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            cleaned = re.sub(r"^```json|```$", "", obj.strip(), flags=re.MULTILINE).strip()
            return json.loads(cleaned)
    if hasattr(obj, "raw"):
        return ensure_dict(obj.raw)
    raise ValueError("Unexpected CrewAI output format")

def clean_json_output(output: str):
    return output.strip().replace("```json", "").replace("```", "")

# ===============================================
# STEP 5: Task
# ===============================================
itinerary_task = Task(
    description="Plan a detailed travel itinerary for {destination}.",
    expected_output="""
        Return ONLY valid JSON matching this schema:
        {
          "name": "string",
          "day_plans": [
            {
              "date": "YYYY-MM-DD",
              "activities": [
                {
                  "name": "string",
                  "location": "string",
                  "description": "string",
                  "date": "YYYY-MM-DD",
                  "cuisine": "string",
                  "why_its_suitable": "string",
                  "reviews": [],
                  "rating": 4.5
                }
              ],
              "restaurants": ["string"],
              "flight": [
                {
                  "airline": "string",
                  "flight_number": "string",
                  "departure": "YYYY-MM-DD HH:MM",
                  "arrival": "YYYY-MM-DD HH:MM",
                  "price": "string"
                }
              ]
            }
          ],
          "hotel": "string"
        }
        ⚠️ Rules:
        - Cover exactly {trip_duration} days.
        - Use {start_date} as Day 1 and increment dates sequentially.
        - Each day must include at least 2 activities and 1 restaurant.
        - Day 1 MUST include at least 2 flight options from {origin} → {destination}.
        - Last day MUST include at least 2 flight options from {destination} → {origin}.
        - Make flights look realistic (major airlines, plausible times, sample prices).
        - Output JSON only (no markdown, no text).
    """,
    agent=itinerary_compiler,
    output_json=Itinerary,
    output_parser=clean_json_output
)

# ===============================================
# STEP 6: Crew
# ===============================================
crew = Crew(
    agents=[itinerary_compiler],
    tasks=[itinerary_task],
    process=Process.sequential,
    verbose=True
)

# ===============================================
# STEP 7: Gradio Interface
# ===============================================
def generate_itinerary(origin, destination, trip_duration, start_date):
    try:
        inputs = {
            "origin": origin,
            "destination": destination,
            "trip_duration": trip_duration,
            "start_date": start_date,
        }
        result = crew.kickoff(inputs=inputs)
        result = ensure_dict(result)

        # Pretty Markdown
        output = f"### ✈️ {result['name']}\n\n**Hotel:** {result['hotel']}\n\n"
        for day in result["day_plans"]:
            output += f"#### 📅 {day['date']}\n"
            if day.get("flight"):
                output += "**✈️ Flight Options:**\n"
                for f in day["flight"]:
                    output += f"- {f['airline']} {f['flight_number']} | {f['departure']} → {f['arrival']} | {f['price']}\n"
            output += "\n**Activities:**\n"
            for act in day["activities"]:
                output += f"- {act['name']} ({act['location']}) ⭐ {act.get('rating','N/A')}\n  {act['description']}\n"
            output += "\n**Restaurants:**\n"
            for rest in day["restaurants"]:
                output += f"- {rest}\n"
            output += "\n---\n"
        return output

    except Exception as e:
        return f"⚠️ Could not generate itinerary. Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("##  Travel Itinerary Planner with Airline Recommendations")
    with gr.Row():
        origin = gr.Textbox(label="Origin", value="New York, JFK")
        destination = gr.Textbox(label="Destination", value="Paris")
    with gr.Row():
        trip_duration = gr.Textbox(label="Trip Duration", value="7 days")
        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2025-10-01")
    btn = gr.Button("Generate Itinerary")
    output = gr.Markdown()

    btn.click(
        fn=generate_itinerary,
        inputs=[origin, destination, trip_duration, start_date],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
