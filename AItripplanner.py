# ========================
# 1. Setup
# ========================
import os, json, re, warnings
import gradio as gr
warnings.filterwarnings("ignore", category=DeprecationWarning)

# API Keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, conlist
from typing import List, Optional

# Vector DB + Embeddings
#!pip install chromadb sentence-transformers
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("user_preferences")

# ========================
# 2. Schemas
# ========================
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

# ========================
# 3. Agent
# ========================
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

# ========================
# 4. Helpers
# ========================
def ensure_dict(obj):
    """Ensure output is parsed JSON"""
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            cleaned = re.sub(r"^```json|```$", "", obj.strip(), flags=re.MULTILINE).strip()
            return json.loads(cleaned)
    return obj

def clean_json_output(output: str):
    return output.strip().replace("```json", "").replace("```", "")

# Store query in vector DB
def log_user_query(user_id, query, structured_output=None):
    embedding = embedder.encode([query])[0].tolist()
    metadata = {"user_id": user_id}
    if structured_output:
        metadata["structured_output"] = structured_output
    collection.add(
        ids=[f"{user_id}_{hash(query)}"],
        embeddings=[embedding],
        documents=[query],
        metadatas=[metadata]
    )

# Retrieve preferences for this user
def get_user_preferences(user_id, new_query, top_k=3):
    embedding = embedder.encode([new_query])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={"user_id": user_id}
    )
    if results["documents"]:
        return " | ".join(results["documents"][0])
    return "No past preferences found."

# ========================
# 5. Task
# ========================
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
        - Consider user preferences: {user_preferences}.
        - Output JSON only (no markdown, no text).
    """,
    agent=itinerary_compiler,
    output_parser=clean_json_output
)

# ========================
# 6. Crew
# ========================
crew = Crew(
    agents=[itinerary_compiler],
    tasks=[itinerary_task],
    process=Process.sequential,
    verbose=True
)

# ========================
# 7. Gradio Interface
# ========================
def generate_itinerary(user_id, origin, destination, trip_duration, start_date, user_preferences):
    query_text = f"{origin} to {destination}, {trip_duration} starting {start_date}, prefs={user_preferences}"
    prefs_history = get_user_preferences(user_id, query_text)

    inputs = {
        "origin": origin,
        "destination": destination,
        "trip_duration": trip_duration,
        "start_date": start_date,
        "user_preferences": prefs_history + " | " + user_preferences
    }

    try:
        result = crew.kickoff(inputs=inputs)
        print("🔎 Raw CrewAI Result:", result)

        # ✅ Extract text safely from CrewOutput
        if hasattr(result, "raw_output"):
            raw_text = result.raw_output
        elif hasattr(result, "output"):
            raw_text = result.output
        else:
            raw_text = str(result)

        # Clean + parse JSON
        cleaned = clean_json_output(raw_text)
        result_dict = ensure_dict(cleaned)

        # ✅ Validate with Pydantic
        itinerary = Itinerary(**result_dict)

        # Store structured itinerary
        log_user_query(user_id, query_text, structured_output=json.dumps(result_dict))

        # Pretty Markdown
        output = f"### ✈️ {itinerary.name}\n\n**Hotel:** {itinerary.hotel}\n\n"
        for day in itinerary.day_plans:
            output += f"#### 📅 {day.date}\n"
            if day.flight:
                output += "**✈️ Flight Options:**\n"
                for f in day.flight:
                    output += f"- {f.airline} {f.flight_number} | {f.departure} → {f.arrival} | {f.price}\n"
            output += "\n**Activities:**\n"
            for act in day.activities:
                output += f"- {act.name} ({act.location}) ⭐ {act.rating}\n  {act.description}\n"
            output += "\n**Restaurants:**\n"
            for rest in day.restaurants:
                output += f"- {rest}\n"
            output += "\n---\n"
        return output

    except Exception as e:
        return f"⚠️ Could not generate itinerary. Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("##  Travel Itinerary Planner with Airline Recommendations + Personalization")
    with gr.Row():
        user_id = gr.Textbox(label="User ID", value="user123")
    with gr.Row():
        origin = gr.Textbox(label="Origin", value="New York, JFK")
        destination = gr.Textbox(label="Destination", value="Paris")
    with gr.Row():
        trip_duration = gr.Textbox(label="Trip Duration", value="3 days")
        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2025-10-01")
    with gr.Row():
        user_preferences = gr.Textbox(label="User Preferences", value="I like museums and Italian food")
    btn = gr.Button("Generate Itinerary")
    output = gr.Markdown()

    btn.click(
        fn=generate_itinerary,
        inputs=[user_id, origin, destination, trip_duration, start_date, user_preferences],
        outputs=output
    )



if __name__ == "__main__":
    demo.launch()

