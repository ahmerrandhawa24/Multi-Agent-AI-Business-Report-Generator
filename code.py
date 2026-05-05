#%%
# ============================================================
# PHASE 1 — Test all imports and connections
# ============================================================

# -- Imports --
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from groq import Groq

load_dotenv()

print("✅ All imports successful")

# ============================================================
# PHASE 1 — Test Groq connection
# ============================================================

from config import GROQ_API_KEY, LLM_MODEL

client   = Groq(api_key=GROQ_API_KEY)
response = client.chat.completions.create(
    model    = LLM_MODEL,
    messages = [{"role": "user", "content": "Say hello in one sentence."}]
)

print(f"✅ Groq working: {response.choices[0].message.content}")

# ============================================================
# PHASE 1 — Test CrewAI LLM setup
# ============================================================

from crewai import LLM

llm = LLM(
    model    = f"groq/{LLM_MODEL}",
    api_key  = GROQ_API_KEY
)

print(f"✅ CrewAI LLM ready: groq/{LLM_MODEL}")

# ============================================================
# PHASE 1 — Test single agent (no tools yet)
# ============================================================

test_agent = Agent(
    role  = "Test Agent",
    goal  = "Confirm the system is working",
    backstory = "You are a test agent checking if CrewAI is set up correctly.",
    llm   = llm,
    verbose = True
)

test_task = Task(
    description = "Say hello and confirm you are working correctly in two sentences.",
    expected_output = "A two sentence confirmation message.",
    agent = test_agent
)

test_crew = Crew(
    agents = [test_agent],
    tasks  = [test_task],
    verbose = True
)

result = test_crew.kickoff()

print(f"\n✅ CrewAI working!")
print(f"Agent output: {result}")

print("\n🎉 Phase 1 complete — ready for Phase 2!")
# %%
# ============================================================
# PHASE 2 - Imports
# ============================================================

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from config import GROQ_API_KEY, LLM_MODEL
from crewai import LLM

print("✅ Phase 2 imports done")
# %%

# ============================================================
# PHASE 2 - LLM Setup
# ============================================================

llm = LLM(
    model   = f"groq/{LLM_MODEL}",
    api_key = GROQ_API_KEY
)

print(f"✅ LLM ready: groq/{LLM_MODEL}")
# %%
# ============================================================
# PHASE 2 - Create Research, Analyst, Writer agents
# ============================================================

# --- Agent 1 — Research Agent ---
research_agent = Agent(
    role = "Senior Research Specialist",
    goal = "Find comprehensive and accurate information about any given topic",
    backstory = """You are an expert researcher with years of experience
    gathering data from multiple sources. You are thorough, accurate,
    and always provide well-structured research findings.""",
    llm     = llm,
    verbose = True
)

# --- Agent 2 — Analyst Agent ---
analyst_agent = Agent(
    role = "Business Analyst",
    goal = "Analyze research data and extract key insights, trends and risks",
    backstory = """You are a seasoned business analyst who specializes in
    turning raw research data into actionable insights. You identify
    patterns, trends, opportunities and risks from any dataset.""",
    llm     = llm,
    verbose = True
)

# --- Agent 3 — Writer Agent ---
writer_agent = Agent(
    role = "Professional Report Writer",
    goal = "Transform insights into a professional, well-structured business report",
    backstory = """You are an expert business writer who creates clear,
    professional reports for executives and stakeholders. Your reports
    are always well-structured, insightful and easy to understand.""",
    llm     = llm,
    verbose = True
)

print("✅ All 3 agents created")
print("   - Research Agent")
print("   - Analyst Agent")
print("   - Writer Agent")


#%%
# ============================================================
# PHASE 2 - Create tasks
# Output of each task feeds into the next agent
# ============================================================

topic = "smartphone market in Pakistan"

# --- Task 1 — Research ---
research_task = Task(
    description = f"""Research the following topic thoroughly:
    Topic: {topic}

    Your research must cover:
    1. Current market size and key players
    2. Recent trends and developments
    3. Consumer behavior and preferences
    4. Major challenges in this market
    5. Growth opportunities

    Provide detailed, factual information for each point.""",

    expected_output = """A comprehensive research report with 5 sections:
    market overview, key players, trends, challenges, and opportunities.
    Each section should have at least 3-4 bullet points.""",

    agent = research_agent
)

# --- Task 2 — Analysis ---
analyst_task = Task(
    description = f"""Analyze the research data provided about {topic}.

    From the research extract:
    1. Top 3 key insights
    2. Top 3 market trends
    3. Top 3 risks
    4. Top 3 opportunities
    5. Overall market sentiment (positive/negative/neutral)

    Be specific and data-driven in your analysis.""",

    expected_output = """A structured analysis with 5 sections:
    key insights, trends, risks, opportunities, and market sentiment.
    Each section should have exactly 3 bullet points.""",

    agent  = analyst_agent,
    context = [research_task]  # gets research output as input
)

# --- Task 3 — Report Writing ---
writer_task = Task(
    description = f"""Write a professional business report about {topic}.

    Using the research and analysis provided, create a report with:
    1. Executive Summary (2-3 sentences)
    2. Market Overview
    3. Key Findings
    4. Risks and Challenges
    5. Opportunities and Recommendations
    6. Conclusion

    The report should be professional, clear and suitable for business executives.""",

    expected_output = """A complete professional business report with all 6 sections,
    written in formal business language, approximately 400-500 words.""",

    agent   = writer_agent,
    context = [research_task, analyst_task]  # gets both outputs
)

print("✅ All 3 tasks created")
print(f"   Topic: {topic}")
print("   Research Task → Analyst Task → Writer Task")

#%%
# ============================================================
# PHASE 2 - Assemble and run the crew
# ============================================================

crew = Crew(
    agents  = [research_agent, analyst_agent, writer_agent],
    tasks   = [research_task, analyst_task, writer_task],
    process = Process.sequential,  # runs in order 1→2→3
    verbose = True
)

print("🚀 Starting crew...\n")
print(f"Topic: {topic}\n")
print("="*60)

result = crew.kickoff()

print("\n" + "="*60)
print("✅ Crew completed!")
print("\n📄 FINAL REPORT:")
print("="*60)
print(result)
# %%
# ============================================================
# PHASE 2 - Save report to outputs folder
# ============================================================

import os
from datetime import datetime
from config import OUTPUT_FOLDER

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename  = f"{OUTPUT_FOLDER}/report_{timestamp}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"TOPIC: {topic}\n")
    f.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    f.write(str(result))

print(f"✅ Report saved: {filename}")
print(f"\n🎉 Phase 2 complete!")


# %%
# ============================================================
# PHASE 3 - Imports
# ============================================================

from crewai import Agent, Task, Crew, Process
from crewai import LLM
from config import GROQ_API_KEY, LLM_MODEL
import os
from datetime import datetime

print("✅ Phase 3 imports done")
# %%
# ============================================================
# PHASE 3 - LLM setup
# ============================================================

llm = LLM(
    model   = f"groq/{LLM_MODEL}",
    api_key = GROQ_API_KEY
)

print(f"✅ LLM ready: groq/{LLM_MODEL}")
# %%# ============================================================
# PHASE 3 - Create all agents including Manager
# ============================================================

# --- Manager Agent ---
manager_agent = Agent(
    role = "Project Manager",
    goal = """Manage and coordinate the research, analysis and
    writing team to produce the best possible business report.
    Break down the topic into clear steps and assign them
    to the right team members.""",
    backstory = """You are an experienced project manager who
    has led hundreds of business intelligence projects. You
    know exactly how to break down complex topics, assign
    the right tasks to the right people, and ensure the
    final output meets the highest standards. You always
    review work quality before accepting it.""",
    llm     = llm,
    allow_delegation = False,
    verbose = True
)

# --- Research Agent ---
research_agent = Agent(
    role = "Senior Research Specialist",
    goal = "Find comprehensive and accurate information about any given topic",
    backstory = """You are an expert researcher with years of
    experience gathering data from multiple sources. You are
    thorough, accurate, and always provide well-structured
    research findings.""",
    llm     = llm,
    allow_delegation = False,
    verbose = True
)

# --- Analyst Agent ---
analyst_agent = Agent(
    role = "Business Analyst",
    goal = "Analyze research data and extract key insights, trends and risks",
    backstory = """You are a seasoned business analyst who
    specializes in turning raw research data into actionable
    insights. You identify patterns, trends, opportunities
    and risks from any dataset.""",
    llm     = llm,
    allow_delegation = False,
    verbose = True
)

# --- Writer Agent ---
writer_agent = Agent(
    role = "Professional Report Writer",
    goal = "Transform insights into a professional business report",
    backstory = """You are an expert business writer who creates
    clear, professional reports for executives and stakeholders.
    Your reports are always well-structured and easy to understand.""",
    llm     = llm,
    allow_delegation = False,
    verbose = True
)

print("✅ All 4 agents created")
print("   - Manager Agent  ← NEW")
print("   - Research Agent")
print("   - Analyst Agent")
print("   - Writer Agent")

# %%
# ============================================================
# PHASE 3 - Create tasks
# Manager oversees all — assigns dynamically
# ============================================================

topic = "electric vehicle market in Pakistan"

# --- Task 1 — Manager plans ---
planning_task = Task(
    description = f"""You are the project manager for this business report:
    Topic: {topic}

    Your job:
    1. Break this topic into 3 clear research questions
    2. Define what the research agent should find
    3. Define what the analyst should focus on
    4. Define the structure for the final report
    5. Set quality standards for the final output

    Be specific and detailed in your plan.""",

    expected_output = """A detailed project plan with:
    - 3 specific research questions
    - Research instructions for the research agent
    - Analysis instructions for the analyst
    - Report structure for the writer
    - Quality checklist with 5 items""",

    agent = manager_agent
)

# --- Task 2 — Research based on manager plan ---
research_task = Task(
    description = f"""Following the project manager's plan, research:
    Topic: {topic}

    Cover these areas thoroughly:
    1. Current market size and valuation
    2. Key players and their market share
    3. Latest trends and developments in 2024
    4. Consumer adoption rates and behavior
    5. Government policies and regulations
    6. Infrastructure challenges and opportunities

    Use the manager's plan as your guide.""",

    expected_output = """Detailed research findings covering all 6 areas.
    Each area should have 3-4 data points or facts.
    Include specific numbers, percentages where possible.""",

    agent   = research_agent,
    context = [planning_task]
)

# --- Task 3 — Analysis ---
analyst_task = Task(
    description = f"""Analyze the research about {topic}.

    Following the manager's instructions:
    1. Extract top 5 key insights
    2. Identify top 3 market trends with evidence
    3. List top 3 critical risks with impact level
    4. List top 3 opportunities with potential value
    5. Give SWOT summary (2 points each)
    6. Rate overall market attractiveness (1-10) with justification

    Be analytical and data-driven.""",

    expected_output = """Structured analysis with all 6 sections completed.
    SWOT summary included. Market attractiveness score with reasoning.""",

    agent   = analyst_agent,
    context = [planning_task, research_task]
)

# --- Task 4 — Final report ---
writer_task = Task(
    description = f"""Write the final business report about {topic}.

    Following the manager's quality standards, create:
    1. Executive Summary (3-4 sentences)
    2. Market Overview (key facts and figures)
    3. Key Findings (top insights from analysis)
    4. SWOT Analysis (formatted table style)
    5. Risks and Challenges
    6. Opportunities and Recommendations
    7. Conclusion and Outlook

    Make it professional, clear and executive-ready.
    Minimum 500 words.""",

    expected_output = """Complete professional business report with all 7
    sections. Well formatted, minimum 500 words, executive-ready quality.""",

    agent   = writer_agent,
    context = [planning_task, research_task, analyst_task]
)

print("✅ All 4 tasks created")
print(f"   Topic: {topic}")
print("   Planning → Research → Analysis → Report")


#%%
# ============================================================
# PHASE 3 - Run crew (sequential with manager review)
# ============================================================
crew = Crew(
    agents  = [research_agent, analyst_agent, writer_agent],
    tasks   = [planning_task, research_task, analyst_task, writer_task],
    process = Process.sequential,
    verbose = True
)
print(f"🚀 Starting managed crew...")
print(f"   Topic  : {topic}")
print(f"   Process: Sequential with Manager Planning")
print("="*60 + "\n")
result = crew.kickoff()
print("\n" + "="*60)
print("✅ Managed crew completed!")
print("\n📄 FINAL REPORT:")
print("="*60)
print(result)

# %%
# ============================================================
# PHASE 3 - Save report
# ============================================================
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename  = f"phase3_report_{timestamp}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"TOPIC    : {topic}\n")
    f.write(f"PROCESS  : Sequential with Manager Planning\n")
    f.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    f.write(str(result))

print(f"✅ Report saved: {filename}")
print(f"\n🎉 Phase 3 complete — Manager Agent working!")
# %%
# ============================================================
# PHASE 4 - Imports
# ============================================================

from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai_tools import (
    FileReadTool,
    DirectoryReadTool,
)
from config import GROQ_API_KEY, LLM_MODEL
import os
from datetime import datetime

print("✅ Phase 4 imports done")
# %%

# ============================================================
# PHASE 4 - LLM setup with smaller model
# llama-3.1-8b-instant = faster + higher rate limits
# ============================================================

llm = LLM(
    model       = "groq/llama-3.1-8b-instant",
    api_key     = GROQ_API_KEY,
    max_tokens  = 800,
    temperature = 0.3,
    max_retries = 3
)

print("✅ LLM ready: llama-3.1-8b-instant")
print("   Faster model — higher rate limits")
# %%
# ============================================================
# PHASE 4 - Setup tools
# ============================================================

from crewai.tools import BaseTool
from duckduckgo_search import DDGS

# --- Tool 1 — Web search ---
class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = """Search the web for current information
    about any topic. Use this to find real-time data,
    news and market information."""

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if not results:
                    return "No results found."
                output = f"Search results for: {query}\n\n"
                for i, r in enumerate(results, 1):
                    output += f"{i}. {r['title']}\n"
                    output += f"   {r['body']}\n\n"
                return output
        except Exception as e:
            return f"Search error: {str(e)}"


# --- Tool 2 — Calculator ---
class CalculatorTool(BaseTool):
    name: str = "Calculator Tool"
    description: str = """Perform mathematical calculations.
    Use for market size calculations, growth rates,
    percentages and numerical analysis."""

    def _run(self, expression: str) -> str:
        try:
            allowed = {
                "__builtins__": {},
                "abs": abs, "round": round,
                "min": min, "max": max,
                "sum": sum, "pow": pow
            }
            result = eval(expression, allowed)
            return f"Result: {expression} = {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"


# --- Instantiate tools ---
web_search_tool = WebSearchTool()
calculator_tool = CalculatorTool()

print("✅ Tools ready:")
print("   - Web Search Tool (DuckDuckGo — free)")
print("   - Calculator Tool")
# %%
# ============================================================
# PHASE 4 - Create agents with tools attached
# ============================================================

# --- Manager Agent ---
manager_agent = Agent(
    role = "Project Manager",
    goal = """Manage and coordinate the research, analysis
    and writing team to produce the best possible
    business report.""",
    backstory = """You are an experienced project manager who
    has led hundreds of business intelligence projects.
    You break down complex topics and set clear quality
    standards for your team.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

# --- Research Agent WITH web search ---
research_agent = Agent(
    role = "Senior Research Specialist",
    goal = """Find comprehensive, accurate and current
    information using web search tools.""",
    backstory = """You are an expert researcher who uses
    web search to find the latest and most accurate data.
    You never rely on outdated information — you always
    search for current facts.""",
    tools            = [web_search_tool],
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

# --- Analyst Agent WITH calculator ---
analyst_agent = Agent(
    role = "Business Analyst",
    goal = """Analyze research data, perform calculations
    and extract key insights with supporting numbers.""",
    backstory = """You are a data-driven business analyst
    who uses calculators and analytical tools to provide
    precise, number-backed insights and recommendations.""",
    tools            = [calculator_tool],
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

# --- Writer Agent ---
writer_agent = Agent(
    role = "Professional Report Writer",
    goal = "Transform insights into a professional business report",
    backstory = """You are an expert business writer who creates
    clear, professional reports for executives. Your reports
    are always well-structured and data-backed.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

print("✅ All 4 agents created with tools")
print("   - Manager Agent")
print("   - Research Agent  + Web Search Tool")
print("   - Analyst Agent   + Calculator Tool")
print("   - Writer Agent")
# %%
# ============================================================
# PHASE 4 - Shorter tasks to reduce token usage
# ============================================================

topic = "e-commerce market in Pakistan"

planning_task = Task(
    description = f"""Plan a business report on: {topic}
    Define: 3 research questions, key data needed,
    report structure and quality standards.""",
    expected_output = "Brief project plan with research questions and report structure.",
    agent = manager_agent
)

research_task = Task(
    description = f"""Use Web Search Tool to research: {topic}
    Search for:
    1. "{topic} market size 2024"
    2. "{topic} top companies"
    3. "{topic} latest trends"
    Summarize findings from each search briefly.""",
    expected_output = "Research summary with market size, key players and trends.",
    agent   = research_agent,
    context = [planning_task]
)

analyst_task = Task(
    description = f"""Analyze research about {topic}.
    Use Calculator Tool for growth rate calculations.
    Provide: 3 key insights, 3 risks, 3 opportunities,
    SWOT (2 points each), market score 1-10.""",
    expected_output = "Analysis with insights, SWOT and market score.",
    agent   = analyst_agent,
    context = [planning_task, research_task]
)

writer_task = Task(
    description = f"""Write a business report on {topic}.
    Include: Executive Summary, Market Overview,
    Key Findings, SWOT, Risks, Opportunities, Conclusion.
    Keep it concise — around 400 words.""",
    expected_output = "Professional business report — 400 words with all sections.",
    agent   = writer_agent,
    context = [planning_task, research_task, analyst_task]
)

print("✅ Shorter tasks created — optimized for free tier")
print(f"   Topic: {topic}")
# %%
# ============================================================
# PHASE 4 - Run crew with rate limit protection
# ============================================================

import time

# Small delay to avoid rate limits
original_kickoff = crew.kickoff

crew = Crew(
    agents  = [manager_agent, research_agent,
               analyst_agent, writer_agent],
    tasks   = [planning_task, research_task,
               analyst_task, writer_task],
    process = Process.sequential,
    verbose = True
)

print(f"🚀 Starting crew with tools...")
print(f"   Topic  : {topic}")
print(f"   Note   : Delays added to avoid rate limits")
print("="*60 + "\n")

# Wait 10 seconds before starting to reset rate limit window
print("⏳ Waiting 10s to reset rate limit window...")
time.sleep(10)

result = crew.kickoff()

print("\n" + "="*60)
print("✅ Crew with tools completed!")
print("\n📄 FINAL REPORT:")
print("="*60)
print(result)
# %%
# ============================================================
# PHASE 4 - Save report
# ============================================================
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename  = f"phase4_report_{timestamp}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"TOPIC    : {topic}\n")
    f.write(f"TOOLS    : Web Search + Calculator\n")
    f.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    f.write(str(result))

print(f"✅ Report saved: {filename}")
print(f"\n🎉 Phase 4 complete — Agents using real tools!")
# %%
# ============================================================
# PHASE 5 - Imports
# ============================================================

from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import json
from datetime import datetime
from config import GROQ_API_KEY, OUTPUT_FOLDER

print("✅ Phase 5 imports done")
# %%
# ============================================================
# PHASE 5 - Back to Groq — but agents WITHOUT tools
# Tools will be called manually before crew runs
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = LLM(
    model       = "groq/llama-3.3-70b-versatile",
    api_key     = GROQ_API_KEY,
    max_tokens  = 600,
    temperature = 0.3,
    max_retries = 5
)

print("✅ LLM ready: Groq llama-3.3-70b")
# %%
# ============================================================
# PHASE 5 - Long-term memory using FAISS
# ============================================================

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from datetime import datetime
from config import OUTPUT_FOLDER

#%%
MEMORY_FOLDER     = "memory"
MEMORY_INDEX_PATH = f"{MEMORY_FOLDER}/memory.faiss"
MEMORY_DATA_PATH  = f"{MEMORY_FOLDER}/memory.pkl"

os.makedirs(MEMORY_FOLDER, exist_ok=True)

print("⏳ Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

def save_to_memory(topic, report_text):
    if os.path.exists(MEMORY_INDEX_PATH):
        index = faiss.read_index(MEMORY_INDEX_PATH)
        with open(MEMORY_DATA_PATH, "rb") as f:
            memory_data = pickle.load(f)
    else:
        index       = faiss.IndexFlatL2(384)
        memory_data = []

    embedding = embedder.encode([report_text]).astype("float32")
    index.add(embedding)
    memory_data.append({
        "topic"     : topic,
        "report"    : report_text,
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id"        : len(memory_data)
    })

    faiss.write_index(index, MEMORY_INDEX_PATH)
    with open(MEMORY_DATA_PATH, "wb") as f:
        pickle.dump(memory_data, f)

    print(f"✅ Saved to memory: {topic}")
    print(f"   Total memories : {len(memory_data)}")


def search_memory(query, top_k=2):
    if not os.path.exists(MEMORY_INDEX_PATH):
        return "No previous reports found in memory."

    index = faiss.read_index(MEMORY_INDEX_PATH)
    with open(MEMORY_DATA_PATH, "rb") as f:
        memory_data = pickle.load(f)

    if len(memory_data) == 0:
        return "Memory is empty."

    query_emb      = embedder.encode([query]).astype("float32")
    distances, ids = index.search(
        query_emb,
        min(top_k, len(memory_data))
    )

    results = []
    for idx in ids[0]:
        mem = memory_data[idx]
        results.append(
            f"Past Report #{mem['id']+1}\n"
            f"Topic    : {mem['topic']}\n"
            f"Date     : {mem['timestamp']}\n"
            f"Summary  : {mem['report'][:300]}...\n"
        )
    return "\n---\n".join(results)


def get_all_memories():
    if not os.path.exists(MEMORY_DATA_PATH):
        return []
    with open(MEMORY_DATA_PATH, "rb") as f:
        return pickle.load(f)


print("✅ Long-term memory system ready")
print(f"   Storage: {MEMORY_FOLDER}/")
# %%
# ============================================================
# PHASE 5 - Short-term memory
# ============================================================

class ShortTermMemory:

    def __init__(self, max_turns=5):
        self.history   = []
        self.max_turns = max_turns

    def add(self, role, content):
        self.history.append({
            "role"      : role,
            "content"   : content,
            "timestamp" : datetime.now().strftime("%H:%M:%S")
        })
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_context(self):
        if not self.history:
            return "No previous conversation."
        context = "Session history:\n"
        for turn in self.history:
            context += f"[{turn['timestamp']}] {turn['role']}: "
            context += f"{turn['content'][:200]}\n"
        return context

    def clear(self):
        self.history = []
        print("✅ Short-term memory cleared")

    def show(self):
        print(f"\n📝 Short-term memory ({len(self.history)} turns):")
        for turn in self.history:
            print(f"  [{turn['timestamp']}] {turn['role']}")
            print(f"  {turn['content'][:150]}...")
            print()


session_memory = ShortTermMemory(max_turns=5)
print("✅ Short-term memory ready")
# %%
# ============================================================
# PHASE 5 - Tools
# ============================================================

from crewai.tools import BaseTool
from duckduckgo_search import DDGS

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = """Search web for current market
    information, news and data."""

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
                if not results:
                    return "No results found."
                output = f"Results for: {query}\n\n"
                for i, r in enumerate(results, 1):
                    output += f"{i}. {r['title']}\n"
                    output += f"   {r['body']}\n\n"
                return output
        except Exception as e:
            return f"Search error: {str(e)}"


class MemorySearchTool(BaseTool):
    name: str = "Memory Search Tool"
    description: str = """Search previous reports stored
    in memory. Use when asked to compare with past work."""

    def _run(self, query: str) -> str:
        return search_memory(query)


class CalculatorTool(BaseTool):
    name: str = "Calculator Tool"
    description: str = """Perform mathematical calculations
    for market projections and analysis."""

    def _run(self, expression: str) -> str:
        try:
            allowed = {
                "__builtins__": {},
                "abs": abs, "round": round,
                "min": min, "max": max,
                "sum": sum, "pow": pow
            }
            result = eval(expression, allowed)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error: {str(e)}"


web_search_tool = WebSearchTool()
memory_tool     = MemorySearchTool()
calculator_tool = CalculatorTool()

print("✅ All tools ready:")
print("   - Web Search Tool")
print("   - Memory Search Tool ← NEW")
print("   - Calculator Tool")
# %%
# ============================================================
# PHASE 5 - Agents WITHOUT tools
# Web search runs manually before crew starts
# ============================================================

from crewai import Agent

manager_agent = Agent(
    role = "Project Manager",
    goal = "Plan business reports and coordinate team.",
    backstory = "Experienced project manager who plans research clearly.",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

research_agent = Agent(
    role = "Senior Research Specialist",
    goal = "Summarize and organize provided research data.",
    backstory = "Expert researcher who organizes data into clear findings.",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

analyst_agent = Agent(
    role = "Business Analyst",
    goal = "Analyze research data and extract key insights.",
    backstory = "Data-driven analyst who finds patterns and insights.",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

writer_agent = Agent(
    role = "Professional Report Writer",
    goal = "Write professional business reports.",
    backstory = "Expert writer who creates executive-ready reports.",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

print("✅ Agents created — no tools attached")
print("   Web search will run manually before crew")


# %%
# ============================================================
# PHASE 5 - Run with manual web search + memory
# ============================================================

import time
from crewai import Task, Crew, Process
from duckduckgo_search import DDGS

topic = "fintech market in Pakistan"

# Step 1 — Manual web search before crew runs
print("🔍 Running web search manually...\n")
search_results = ""
try:
    with DDGS() as ddgs:
        results = list(ddgs.text(
            f"{topic} market size 2024", max_results=4
        ))
        for r in results:
            search_results += f"- {r['title']}: {r['body']}\n"
    print(f"✅ Search done — {len(results)} results found")
except Exception as e:
    search_results = "Search unavailable — use general knowledge."
    print(f"⚠️ Search failed: {e}")

# Step 2 — Check memory
print("\n🧠 Checking memory...")
past = search_memory(topic)
print(f"   {past[:100]}")

# Step 3 — Add to session memory
session_memory.add("user", f"Analyze: {topic}")

# Step 4 — Create tasks with search data injected
planning_task = Task(
    description = f"""Plan a business report on: {topic}
    List 3 research questions. Be brief.""",
    expected_output = "3 research questions in bullet points.",
    agent = manager_agent
)

research_task = Task(
    description = f"""Organize this research data about {topic}:

    {search_results}

    Summarize into 5 key bullet points.""",
    expected_output = "5 bullet point research summary.",
    agent   = research_agent,
    context = [planning_task]
)

analyst_task = Task(
    description = f"""Analyze the research about {topic}.
    Provide: 3 insights, 2 risks, 2 opportunities.
    Rate market attractiveness 1-10.""",
    expected_output = "Brief analysis with market score.",
    agent   = analyst_agent,
    context = [research_task]
)

writer_task = Task(
    description = f"""Write a short business report on {topic}.
    Sections: Executive Summary, Key Findings,
    Risks, Opportunities, Conclusion.
    Maximum 350 words.""",
    expected_output = "350 word professional business report.",
    agent   = writer_agent,
    context = [analyst_task]
)

# Step 5 — Run crew with delays
print("\n🚀 Starting crew...")
print(f"   Topic: {topic}")
print("="*60 + "\n")

results = {}
tasks_agents = [
    ("Planning",  planning_task,  manager_agent),
    ("Research",  research_task,  research_agent),
    ("Analysis",  analyst_task,   analyst_agent),
    ("Writing",   writer_task,    writer_agent),
]

for name, task, agent in tasks_agents:
    print(f"⏳ Waiting 20s before {name}...")
    time.sleep(20)
    print(f"🤖 Running {name} agent...")

    mini_crew = Crew(
        agents  = [agent],
        tasks   = [task],
        process = Process.sequential,
        verbose = False
    )

    try:
        result      = mini_crew.kickoff()
        results[name] = str(result)
        print(f"✅ {name} done")
        print(f"   {str(result)[:120]}...")
    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
        results[name] = "Task failed"

# Step 6 — Save to memory
final_report = results.get("Writing", "No report")
save_to_memory(topic, final_report)
session_memory.add("assistant", final_report[:300])

print("\n" + "="*60)
print("📄 FINAL REPORT:")
print("="*60)
print(final_report)
# %%
# ============================================================
# PHASE 5 - Test memory works
# ============================================================

print("🧠 Testing memory retrieval...\n")

# Test 1 — search long-term memory
print("Test 1 — Long-term memory search:")
print("-"*40)
memory_result = search_memory("fintech Pakistan")
print(memory_result[:400])

print("\nTest 2 — Short-term session memory:")
print("-"*40)
session_memory.show()

print("\nTest 3 — All stored memories:")
print("-"*40)
all_memories = get_all_memories()
print(f"Total reports in memory: {len(all_memories)}")
for mem in all_memories:
    print(f"  #{mem['id']+1} | {mem['topic']} | {mem['timestamp']}")

print("\n🎉 Phase 5 complete — Memory system working!")
# %%
# ============================================================
# PHASE 6 - Imports
# ============================================================

import pandas as pd
import os
import json
import time
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("✅ Phase 6 imports done")
# %%
# ============================================================
# PHASE 6 - LLM setup
# ============================================================

llm = LLM(
    model       = "groq/llama-3.3-70b-versatile",
    api_key     = GROQ_API_KEY,
    max_tokens  = 600,
    temperature = 0.3,
    max_retries = 5
)

print("✅ LLM ready: llama-3.3-70b")
# %%
# ============================================================
# PHASE 6 - Create sample business dataset
# In real project user uploads their own CSV
# ============================================================

import os
os.makedirs("data", exist_ok=True)

# Sample e-commerce sales data for Pakistan
data = {
    "Month"       : ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "Revenue_PKR" : [1200000, 1350000, 1100000, 1500000, 1800000,
                     1650000, 1900000, 2100000, 1950000, 2300000,
                     2800000, 3200000],
    "Orders"      : [450, 520, 410, 580, 690, 640,
                     720, 810, 760, 890, 1050, 1200],
    "Customers"   : [320, 380, 290, 420, 510, 480,
                     540, 610, 580, 670, 790, 920],
    "Returns"     : [23, 28, 19, 31, 42, 35,
                     38, 44, 41, 48, 56, 63],
    "Category"    : ["Electronics", "Fashion", "Electronics",
                     "Fashion", "Electronics", "Home",
                     "Electronics", "Fashion", "Home",
                     "Electronics", "Fashion", "Electronics"]
}

df = pd.DataFrame(data)
df.to_csv("data/sales_data.csv", index=False)

print("✅ Sample CSV created: data/sales_data.csv")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {list(df.columns)}")
print(f"\n--- Preview ---")
print(df.head())
# %%
# ============================================================
# PHASE 6 - ETL Pipeline
# Extract → Transform → Load
# ============================================================

def extract(filepath):
    """Extract — load raw data from CSV"""
    df = pd.read_csv(filepath)
    print(f"✅ Extracted: {len(df)} rows from {filepath}")
    return df


def transform(df):
    """Transform — clean and enrich data"""

    # Remove duplicates
    df = df.drop_duplicates()

    # Add calculated columns
    df["Avg_Order_Value"]  = (
        df["Revenue_PKR"] / df["Orders"]
    ).round(2)

    df["Return_Rate_Pct"]  = (
        df["Returns"] / df["Orders"] * 100
    ).round(2)

    df["Revenue_Per_Customer"] = (
        df["Revenue_PKR"] / df["Customers"]
    ).round(2)

    # Add month number for sorting
    month_map = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,
        "May":5,"Jun":6,"Jul":7,"Aug":8,
        "Sep":9,"Oct":10,"Nov":11,"Dec":12
    }
    df["Month_Num"] = df["Month"].map(month_map)
    df = df.sort_values("Month_Num")

    print("✅ Transformed:")
    print(f"   Added columns: Avg_Order_Value, Return_Rate_Pct, Revenue_Per_Customer")
    return df


def load(df, output_path):
    """Load — save cleaned data"""
    df.to_csv(output_path, index=False)
    print(f"✅ Loaded: cleaned data saved to {output_path}")
    return df


def run_etl(input_path, output_path):
    """Run full ETL pipeline"""
    print("🔄 Running ETL pipeline...")
    print(f"   Input  : {input_path}")
    print(f"   Output : {output_path}\n")

    df = extract(input_path)
    df = transform(df)
    df = load(df, output_path)

    print(f"\n✅ ETL complete")
    return df


# Run ETL
clean_df = run_etl(
    input_path  = "data/sales_data.csv",
    output_path = "data/sales_clean.csv"
)

print(f"\n--- Cleaned Data Preview ---")
print(clean_df[["Month", "Revenue_PKR", "Avg_Order_Value",
                "Return_Rate_Pct"]].to_string(index=False))
# %%
# ============================================================
# PHASE 6 - Generate stats summary
# This gets passed to data agent as context
# ============================================================

def generate_stats(df):
    """Generate statistical summary for data agent"""

    total_revenue    = df["Revenue_PKR"].sum()
    total_orders     = df["Orders"].sum()
    total_customers  = df["Customers"].sum()
    avg_order_value  = df["Avg_Order_Value"].mean().round(2)
    avg_return_rate  = df["Return_Rate_Pct"].mean().round(2)
    best_month       = df.loc[df["Revenue_PKR"].idxmax(), "Month"]
    worst_month      = df.loc[df["Revenue_PKR"].idxmin(), "Month"]

    # Growth rate
    first_revenue    = df.iloc[0]["Revenue_PKR"]
    last_revenue     = df.iloc[-1]["Revenue_PKR"]
    growth_rate      = ((last_revenue - first_revenue)
                        / first_revenue * 100).round(2)

    # Category breakdown
    cat_revenue = df.groupby("Category")["Revenue_PKR"].sum()

    stats = f"""
BUSINESS DATA SUMMARY
=====================
Total Revenue    : PKR {total_revenue:,}
Total Orders     : {total_orders:,}
Total Customers  : {total_customers:,}
Avg Order Value  : PKR {avg_order_value:,}
Avg Return Rate  : {avg_return_rate}%
Best Month       : {best_month}
Worst Month      : {worst_month}
Annual Growth    : {growth_rate}%

Revenue by Category:
{cat_revenue.to_string()}

Monthly Trend:
{df[['Month','Revenue_PKR','Orders']].to_string(index=False)}
"""
    return stats


stats_summary = generate_stats(clean_df)
print("✅ Stats summary generated:")
print(stats_summary)
# %%
# ============================================================
# PHASE 6 - Data agent
# Analyzes structured data stats
# ============================================================

data_agent = Agent(
    role = "Senior Data Analyst",
    goal = """Analyze structured business data and extract
    meaningful insights, trends and recommendations.""",
    backstory = """You are an expert data analyst with years
    of experience analyzing business metrics. You excel at
    finding patterns in sales data, identifying growth
    opportunities and spotting warning signs in the numbers.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

writer_agent = Agent(
    role = "Professional Report Writer",
    goal = "Write professional data-driven business reports.",
    backstory = """Expert business writer who transforms
    data analysis into clear executive reports.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

print("✅ Data agent created")
print("✅ Writer agent created")
# %%
# ============================================================
# PHASE 6 - Run data analysis crew
# ============================================================

# Task 1 — Data analysis
data_task = Task(
    description = f"""Analyze this business data carefully:

{stats_summary}

Provide:
1. Top 3 performance insights from the numbers
2. Monthly growth trend analysis
3. Category performance comparison
4. Top 2 warning signs in the data
5. Top 3 recommendations based on data
6. Overall business health score 1-10""",

    expected_output = """Data analysis with 6 sections:
    insights, trends, category analysis, warnings,
    recommendations and health score.""",
    agent = data_agent
)

# Task 2 — Report writing
report_task = Task(
    description = f"""Write a data-driven business report
    based on the analysis provided.

    Include:
    1. Executive Summary
    2. Key Performance Metrics
    3. Growth Analysis
    4. Category Breakdown
    5. Risk Indicators
    6. Recommendations
    7. Conclusion

    Maximum 400 words. Professional business language.""",

    expected_output = "400 word professional data report.",
    agent   = writer_agent,
    context = [data_task]
)

# Run with delay
print("🚀 Running data analysis crew...")
results = {}

for name, task, agent in [
    ("Data Analysis", data_task,   data_agent),
    ("Report Writing", report_task, writer_agent)
]:
    print(f"\n⏳ Waiting 20s before {name}...")
    time.sleep(20)
    print(f"🤖 Running {name}...")

    mini_crew = Crew(
        agents  = [agent],
        tasks   = [task],
        process = Process.sequential,
        verbose = False
    )

    try:
        result        = mini_crew.kickoff()
        results[name] = str(result)
        print(f"✅ {name} done")
        print(f"   {str(result)[:150]}...")
    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
        results[name] = "Failed"

print("\n" + "="*60)
print("📄 DATA REPORT:")
print("="*60)
print(results.get("Report Writing", "No report"))
# %%

# ============================================================
# PHASE 6 - Save report and ETL log
# ============================================================

os.makedirs("outputs", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save report
report_path = f"outputs/data_report_{timestamp}.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"DATA REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    f.write(results.get("Report Writing", "No report"))

# Save ETL log
log_path = f"outputs/etl_log_{timestamp}.json"
etl_log  = {
    "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "input_file"   : "data/sales_data.csv",
    "output_file"  : "data/sales_clean.csv",
    "rows_processed": len(clean_df),
    "status"       : "success"
}
with open(log_path, "w") as f:
    json.dump(etl_log, f, indent=2)

print(f"✅ Report saved : {report_path}")
print(f"✅ ETL log saved: {log_path}")
# %%
# ============================================================
# PHASE 6 - Create Windows Task Scheduler script
# Auto-runs ETL every day at 9 AM — no Airflow needed
# ============================================================

# Create a standalone ETL runner script
etl_script = '''import pandas as pd
import json
import os
from datetime import datetime

def run_daily_etl():
    print(f"ETL started: {datetime.now()}")

    # Extract
    df = pd.read_csv("data/sales_data.csv")

    # Transform
    df["Avg_Order_Value"] = (df["Revenue_PKR"] / df["Orders"]).round(2)
    df["Return_Rate_Pct"] = (df["Returns"] / df["Orders"] * 100).round(2)
    df = df.drop_duplicates()

    # Load
    df.to_csv("data/sales_clean.csv", index=False)

    # Log
    log = {
        "timestamp"     : str(datetime.now()),
        "rows_processed": len(df),
        "status"        : "success"
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/etl_log_latest.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"ETL complete: {len(df)} rows processed")

if __name__ == "__main__":
    run_daily_etl()
'''

with open("etl_runner.py", "w") as f:
    f.write(etl_script)

print("✅ ETL runner script created: etl_runner.py")

# Create Task Scheduler setup instructions
scheduler_script = f'''
@echo off
echo Setting up Windows Task Scheduler for daily ETL...

schtasks /create /tn "DailyETL" /tr "python {os.path.abspath("etl_runner.py")}" /sc daily /st 09:00 /f

echo.
echo Task created successfully!
echo ETL will run every day at 9:00 AM automatically.
pause
'''

with open("setup_scheduler.bat", "w") as f:
    f.write(scheduler_script)

print("✅ Task Scheduler setup created: setup_scheduler.bat")
print("\nTo enable daily auto-run:")
print("   1. Right-click setup_scheduler.bat")
print("   2. Run as Administrator")
print("   3. ETL will run every day at 9 AM automatically")
print("\n🎉 Phase 6 complete — Data Agent + ETL working!")
# %%
# ============================================================
# PHASE 8 - Imports
# ============================================================

from crewai import Agent, Task, Crew, Process, LLM
import os
import time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("✅ Phase 8 imports done")
# %%
# ============================================================
# PHASE 8 - LLM setup
# ============================================================

llm = LLM(
    model       = "groq/llama-3.3-70b-versatile",
    api_key     = GROQ_API_KEY,
    max_tokens  = 600,
    temperature = 0.3,
    max_retries = 5
)

print("✅ LLM ready")
# %%
# ============================================================
# PHASE 8 - Evaluation agent
# Scores report quality and gives improvement feedback
# ============================================================

evaluation_agent = Agent(
    role = "Quality Evaluation Specialist",
    goal = """Evaluate business reports for quality,
    accuracy, completeness and professionalism.
    Provide specific improvement suggestions.""",
    backstory = """You are a senior quality assurance
    specialist with 15 years of experience reviewing
    business reports. You have very high standards and
    always provide constructive, specific feedback.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

print("✅ Evaluation agent created")
# %%
# ============================================================
# PHASE 8 - Critic agent
# Finds weaknesses in research and analysis
# ============================================================

critic_agent = Agent(
    role = "Research Critic",
    goal = """Critically review research and analysis
    findings. Identify gaps, weak arguments and
    missing data points.""",
    backstory = """You are a sharp critical thinker who
    has reviewed thousands of business reports. You
    always find what is missing and what could be
    stronger. You are constructive but thorough.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

print("✅ Critic agent created")
# %%
# ============================================================
# PHASE 8 - Improver agent
# Takes critic feedback and improves the report
# ============================================================

improver_agent = Agent(
    role = "Report Improvement Specialist",
    goal = """Take critic feedback and improve the
    business report to address all weaknesses.""",
    backstory = """You are an expert at taking feedback
    and improving reports. You address every criticism
    specifically and make reports significantly better.""",
    llm              = llm,
    allow_delegation = False,
    verbose          = True
)

print("✅ Improver agent created")
# %%
# ============================================================
# PHASE 8 - Evaluate report quality
# Returns score and feedback
# ============================================================

def evaluate_report(report_text, topic):
    """Run evaluation agent on a report"""

    eval_task = Task(
        description = f"""Evaluate this business report
        on topic: {topic}

        REPORT TO EVALUATE:
        {report_text[:1500]}

        Score these dimensions (1-10 each):
        1. Completeness — are all sections present?
        2. Data quality — are there specific numbers?
        3. Insight depth — are insights meaningful?
        4. Professionalism — is language appropriate?
        5. Actionability — are recommendations clear?

        Then give:
        - Overall score (average of above)
        - Top 3 specific improvement suggestions
        - Pass or Fail (pass = score 7 or above)

        Respond in this exact format:
        Completeness: X/10
        Data Quality: X/10
        Insight Depth: X/10
        Professionalism: X/10
        Actionability: X/10
        Overall Score: X/10
        Verdict: PASS or FAIL
        Improvements:
        1. [specific improvement]
        2. [specific improvement]
        3. [specific improvement]""",

        expected_output = """Evaluation with scores,
        verdict and 3 improvement suggestions.""",
        agent = evaluation_agent
    )

    print("⏳ Waiting 20s before evaluation...")
    time.sleep(20)

    mini_crew = Crew(
        agents  = [evaluation_agent],
        tasks   = [eval_task],
        process = Process.sequential,
        verbose = False
    )

    try:
        result = mini_crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Evaluation failed: {str(e)}"


print("✅ Evaluation function ready")
# %%
# ============================================================
# PHASE 8 - Critique and improve report
# Collaboration loop between critic and improver
# ============================================================

def critique_and_improve(report_text, topic, eval_feedback):
    """Critic finds weaknesses, improver fixes them"""

    # Step 1 — Critic reviews
    critic_task = Task(
        description = f"""Review this business report
        on {topic} and find specific weaknesses:

        REPORT:
        {report_text[:1200]}

        EVALUATION FEEDBACK:
        {eval_feedback[:500]}

        Identify:
        1. Top 3 missing data points
        2. Top 2 weak arguments
        3. What sections need more depth
        4. Any factual concerns

        Be specific and constructive.""",

        expected_output = """Specific critique with
        missing data, weak arguments and improvement areas.""",
        agent = critic_agent
    )

    print("⏳ Waiting 20s before critique...")
    time.sleep(20)

    critic_crew = Crew(
        agents  = [critic_agent],
        tasks   = [critic_task],
        process = Process.sequential,
        verbose = False
    )

    try:
        critique = str(critic_crew.kickoff())
        print("✅ Critique done")
        print(f"   {critique[:150]}...")
    except Exception as e:
        critique = "Critique unavailable"
        print(f"⚠️ Critique failed: {e}")

    # Step 2 — Improver fixes based on critique
    improve_task = Task(
        description = f"""Improve this business report
        based on the critique provided.

        ORIGINAL REPORT:
        {report_text[:1200]}

        CRITIQUE TO ADDRESS:
        {critique[:600]}

        Instructions:
        - Address every criticism specifically
        - Add missing data points where possible
        - Strengthen weak arguments
        - Keep all original good sections
        - Maximum 450 words
        - Professional business language""",

        expected_output = """Improved business report
        addressing all critique points. 450 words max.""",
        agent = improver_agent
    )

    print("⏳ Waiting 20s before improvement...")
    time.sleep(20)

    improve_crew = Crew(
        agents  = [improver_agent],
        tasks   = [improve_task],
        process = Process.sequential,
        verbose = False
    )

    try:
        improved = str(improve_crew.kickoff())
        print("✅ Improvement done")
        return improved, critique
    except Exception as e:
        print(f"⚠️ Improvement failed: {e}")
        return report_text, critique


print("✅ Critique and improve function ready")
# %%
# ============================================================
# PHASE 8 - Full evaluation pipeline
# Evaluate → if fail → critique → improve → re-evaluate
# ============================================================

def run_evaluation_pipeline(report_text, topic, max_retries=2):
    """
    Full quality loop:
    1. Evaluate report
    2. If fail → critique + improve
    3. Re-evaluate improved report
    4. Return best version
    """

    print(f"\n{'='*60}")
    print(f"🔍 EVALUATION PIPELINE")
    print(f"   Topic: {topic}")
    print(f"{'='*60}\n")

    best_report  = report_text
    eval_history = []

    for attempt in range(max_retries):
        print(f"\n📊 Evaluation attempt {attempt + 1}...")

        # Evaluate current report
        eval_result = evaluate_report(best_report, topic)
        eval_history.append(eval_result)

        print(f"\n📋 Evaluation result:")
        print(eval_result)

        # Check if passed
        if "PASS" in eval_result.upper():
            print(f"\n✅ Report PASSED evaluation!")
            break

        print(f"\n❌ Report FAILED — running improvement loop...")

        if attempt < max_retries - 1:
            # Critique and improve
            improved, critique = critique_and_improve(
                best_report, topic, eval_result
            )
            best_report = improved
            print(f"\n✅ Report improved — re-evaluating...")

    return best_report, eval_history


# Test with a sample report
sample_report = """
Executive Summary:
The fintech market in Pakistan is growing rapidly.
Key players include JazzCash and Easypaisa.
The market faces regulatory challenges.

Key Findings:
Mobile payments are increasing.
Digital banking is expanding.

Conclusion:
The market has good potential.
"""

topic = "fintech market in Pakistan"

print("🧪 Testing evaluation pipeline with weak report...\n")
final_report, history = run_evaluation_pipeline(
    sample_report, topic, max_retries=2
)

print(f"\n{'='*60}")
print("📄 FINAL REPORT AFTER EVALUATION:")
print("="*60)
print(final_report)

print(f"\n📊 Evaluation history: {len(history)} attempts")
# %%
# ============================================================
# PHASE 8 - Save evaluation results
# ============================================================

os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save final improved report
report_path = f"outputs/evaluated_report_{timestamp}.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"TOPIC     : {topic}\n")
    f.write(f"GENERATED : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"EVALUATED : Yes — {len(history)} attempts\n")
    f.write("="*60 + "\n\n")
    f.write(final_report)

# Save evaluation log
log_path = f"outputs/eval_log_{timestamp}.json"
import json
eval_log = {
    "topic"     : topic,
    "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "attempts"  : len(history),
    "passed"    : "PASS" in history[-1].upper() if history else False,
    "history"   : history
}
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(eval_log, f, indent=2)

print(f"✅ Evaluated report saved : {report_path}")
print(f"✅ Evaluation log saved   : {log_path}")
print(f"\n🎉 Phase 8 complete!")
print(f"   Evaluation attempts : {len(history)}")
print(f"   Final verdict       : {'PASS ✅' if eval_log['passed'] else 'IMPROVED ✅'}")
# %%
