# ✈️ AI-Powered Travel Itinerary Planner

An AI-driven travel itinerary planner that generates **personalized multi-day trips** with:
- 🛫 **Flight recommendations** (airlines, times, prices)
- 🗺️ **Day-by-day activities** (sightseeing, culture, events)
- 🍴 **Restaurant suggestions** (tailored to user preferences)
- 🏨 **Hotel assignment**

Built using **Groq LLMs + CrewAI agents + Gradio UI** with **ChromaDB personalization**.

---

## 🚀 Features
- **Multi-Agent Orchestration**: CrewAI agents collaborate (Activity Planner, Restaurant Scout, Itinerary Compiler).
- **Structured JSON Outputs**: Validated with Pydantic schemas for consistency.
- **Flight Recommendations**: Day 1 includes flights from origin → destination, last day includes return flights.
- **Vector DB Personalization**:
  - User queries & preferences stored in **ChromaDB**.
  - Learns from history (e.g., museums, Italian food).
- **Interactive Gradio UI**: Input origin, destination, duration, and start date → get instant travel plan.

---

## 🛠️ Tech Stack
- **LLM & Agents**: [Groq LLM](https://groq.com/), [CrewAI](https://github.com/joaomdmoura/crewai)
- **UI**: [Gradio](https://gradio.app/)
- **Persistence**: [ChromaDB](https://www.trychroma.com/), [SentenceTransformers](https://www.sbert.net/)
- **Backend**: Python, Colab/Jupyter environment
- **Validation**: Pydantic models

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/<your-username>/travel-itinerary-planner.git
cd travel-itinerary-planner

# Install dependencies
pip install -r requirements.txt
