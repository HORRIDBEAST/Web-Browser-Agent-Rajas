Web Browser Query Agent
This project is an intelligent web search agent built to fulfill the Ripplica Interview Task. It features a full-stack application with a Next.js frontend and a FastAPI backend. The agent can validate user queries, check for semantically similar past searches in a cache, perform new searches using a robust API, and summarize the findings with a powerful AI model.

Video Demonstration: Link to Video

## ✨ Features

Full-Stack Application: A sleek, responsive Next.js frontend communicates with a powerful Python FastAPI backend.

Intelligent Query Validation: The agent first classifies queries to distinguish between valid search requests and invalid commands (e.g., "walk my dog").

Semantic Caching: Utilizes sentence-transformers to generate vector embeddings for queries. This allows the agent to find and return cached results for semantically similar queries (e.g., "Best places in Delhi" ≈ "Top Delhi attractions").

Robust Web Search: Employs the Tavily Search API for reliable, fast, and structured search results, avoiding the fragility of direct web scraping.

AI-Powered Summarization: Leverages Google's Gemini 1.5 Flash model to provide concise, comprehensive summaries from the search results.

Persistent Storage: Uses an SQLite database (query_agent.db) to store query embeddings, search results, and summaries for efficient caching.

Interactive CLI: A fully functional command-line interface (query_agent.py) for backend testing and direct interaction.

Real-time Frontend: The Next.js UI provides a smooth user experience with loading states, error handling, and a display for cached vs. new results.

Detailed Logging: The FastAPI backend includes comprehensive logging to monitor incoming requests, cache hits/misses, and potential errors.

🏛️ Architecture & Data Flow
The agent is designed with a clear separation between the user interface, the backend logic, and the data layer. The data flows through a series of validation, caching, and processing steps to deliver an accurate and fast response.

Code snippet

graph TD
    subgraph "User Interface (Next.js)"
        A[User Enters Query] --> B[POST Request to /api/search]
        Z[Display Result or Error]
    end

    subgraph "Backend API (FastAPI)"
        B --> C[Receive Request]
        C --> D{1. Is Query Valid?}
        D -- No --> E[Return 400 Error]
        D -- Yes --> F{2. Find Similar Query in DB?}

        subgraph "Path A: Cache Hit"
            F -- Yes --> G[Retrieve Cached Summary from SQLite]
            G --> R([Return JSON Response])
        end

        subgraph "Path B: Cache Miss"
            F -- No --> I[3. Search Web with Tavily API]
            I -- Structured Content --> J[4. Summarize Content with Gemini AI]
            J -- New Summary & Embedding --> K[5. Store Result in SQLite]
            K --> R
        end
    end

    R --> Z
    E --> Z
🛠️ Tech Stack
Component	Technology
Frontend	Next.js, React, Tailwind CSS, Shadcn/ui
Backend	FastAPI, Python 3
AI Summarization	Google Gemini API (gemini-1.5-flash)
Web Search	Tavily Search API
Embeddings	Sentence-Transformers (all-MiniLM-L6-v2)
Database	SQLite
CLI	Click

Export to Sheets
🧠 Engineering Decisions
Web Scraping vs. Search API (Tavily): Initial attempts with direct web scraping (using Playwright on DuckDuckGo/Google) proved to be extremely unreliable due to anti-bot measures, captchas, and frequent HTML structure changes. The decision was made to switch to the Tavily Search API. This provides a more robust, stable, and professional solution by delivering structured JSON data, eliminating the need for fragile CSS selectors and significantly improving the agent's reliability and speed.

Semantic Similarity Threshold: The similarity threshold in query_agent.py was set to 0.85. This value was chosen after testing to strike a balance between correctly identifying similar queries (like "places in Delhi" vs. "Delhi attractions") and preventing false positives (like "places in Delhi" vs. "places in Darjeeling").

Frontend-Backend Separation: The frontend is a standalone Next.js application that communicates with the backend via a well-defined REST API. This separation of concerns allows for independent development, scaling, and maintenance. The frontend's simulateAPICall was replaced with actual fetch requests to ensure it is a true client of the backend service.

AI Model Choice (Gemini 1.5 Flash): Gemini 1.5 Flash was selected for its large context window, fast response times, and excellent summarization capabilities, all available through a generous free tier.

State Management (Frontend): Simple React state management (useState, useEffect) was used, which is sufficient for this application's scope. For larger applications, a more robust solution like Redux or Zustand could be integrated.

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.8+

Node.js and npm (or yarn/pnpm)

A virtual environment tool for Python (like venv)

1. API Keys
You will need API keys from two services:

Google Gemini: Get your key from Google AI Studio.

Tavily AI: Get your free key from the Tavily Dashboard.

2. Backend Setup
The backend is located in the backend/ directory.

Bash

# 1. Navigate to the backend directory
cd backend

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the required Python packages
pip install -r requirements.txt

# 4. Create a .env file in the 'backend' directory
#    and add your API keys
touch .env
Your backend/.env file should look like this:

GOOGLE_API_KEY="your-gemini-api-key-here"
TAVILY_API_KEY="your-tavily-api-key-here"
3. Frontend Setup
The frontend is located in the frontend/ directory.

Bash

# 1. Navigate to the frontend directory from the root
cd frontend

# 2. Install the required Node.js packages
npm install

# 3. The frontend is configured to talk to the backend on port 8000.
#    No .env file is needed for the frontend.
4. Running the Application
You need to run the backend and frontend in two separate terminal windows.

Terminal 1: Start the Backend

Bash

cd backend
source venv/bin/activate
python fastapi_backend.py
The API server will start on http://localhost:8000. You will see logs for incoming requests here.

Terminal 2: Start the Frontend

Bash

cd frontend
npm run dev
The frontend application will be available at http://localhost:3000.

⚙️ Usage
Web App: Open http://localhost:3000 in your browser. Enter a query in the search box and press "Search".

CLI (for testing): You can also run the core logic directly from the command line.

Bash

cd backend
source venv/bin/activate
python query_agent.py interactive
