# ============================================================
# app.py — Multi-Agent AI Business Report Generator
# Phase 7 + 8 + 9 — Complete Streamlit UI
# ============================================================

import streamlit as st
import pandas as pd
import os
import json
import time
import pickle
import faiss
import numpy as np
import fitz
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "AI Business Report Generator",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_llm():
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    return LLM(
        model       = "groq/llama-3.3-70b-versatile",
        api_key     = GROQ_API_KEY,
        max_tokens  = 600,
        temperature = 0.3,
        max_retries = 5
    )

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

llm      = load_llm()
embedder = load_embedder()

# ============================================================
# MEMORY SYSTEM
# ============================================================

MEMORY_FOLDER     = "memory"
MEMORY_INDEX_PATH = f"{MEMORY_FOLDER}/memory.faiss"
MEMORY_DATA_PATH  = f"{MEMORY_FOLDER}/memory.pkl"
os.makedirs(MEMORY_FOLDER, exist_ok=True)
os.makedirs("data",        exist_ok=True)
os.makedirs("outputs",     exist_ok=True)

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


def search_memory(query, top_k=2):
    if not os.path.exists(MEMORY_INDEX_PATH):
        return "No previous reports found."
    index = faiss.read_index(MEMORY_INDEX_PATH)
    with open(MEMORY_DATA_PATH, "rb") as f:
        memory_data = pickle.load(f)
    if not memory_data:
        return "Memory is empty."
    query_emb      = embedder.encode([query]).astype("float32")
    distances, ids = index.search(
        query_emb, min(top_k, len(memory_data))
    )
    results = []
    for idx in ids[0]:
        mem = memory_data[idx]
        results.append(
            f"Past Report: {mem['topic']} "
            f"({mem['timestamp']})\n"
            f"{mem['report'][:200]}..."
        )
    return "\n---\n".join(results)


def get_all_memories():
    if not os.path.exists(MEMORY_DATA_PATH):
        return []
    with open(MEMORY_DATA_PATH, "rb") as f:
        return pickle.load(f)

# ============================================================
# ETL PIPELINE
# ============================================================

def run_etl(df):
    df = df.drop_duplicates()
    if "Revenue_PKR" in df.columns and "Orders" in df.columns:
        df["Avg_Order_Value"] = (
            df["Revenue_PKR"] / df["Orders"]
        ).round(2)
    if "Returns" in df.columns and "Orders" in df.columns:
        df["Return_Rate_Pct"] = (
            df["Returns"] / df["Orders"] * 100
        ).round(2)
    return df

def detect_data_type(df):
    cols = " ".join([str(c).lower() for c in df.columns])
    if any(w in cols for w in ["temp", "humidity", "rain", "wind", "weather", "pressure"]):
        return "WEATHER DATA — write a weather analysis report"
    elif any(w in cols for w in ["revenue", "sales", "orders", "profit", "customer"]):
        return "SALES/BUSINESS DATA — write a business performance report"
    elif any(w in cols for w in ["price", "stock", "market", "volume", "close", "open"]):
        return "FINANCIAL DATA — write a financial analysis report"
    else:
        return "GENERAL DATA — write a data analysis report based strictly on the actual columns"
def generate_stats(df):
    stats = "DATA SUMMARY\n" + "="*40 + "\n"
    stats += f"Rows     : {len(df)}\n"
    stats += f"Columns  : {list(df.columns)}\n\n"
    stats += f"Data Type: {detect_data_type(df)}\n\n"

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        stats += f"{col}:\n"
        stats += f"  Total : {df[col].sum():,.2f}\n"
        stats += f"  Avg   : {df[col].mean():,.2f}\n"
        stats += f"  Max   : {df[col].max():,.2f}\n"
        stats += f"  Min   : {df[col].min():,.2f}\n\n"

    return stats

# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):
    try:
        doc  = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            text += f"\n--- Page {page_num+1} ---\n"
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return f"PDF extraction failed: {str(e)}"

# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query, max_results=4):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            output = ""
            for r in results:
                output += f"- {r['title']}: {r['body']}\n"
            return output
    except Exception as e:
        return f"Search unavailable: {str(e)}"

# ============================================================
# AGENTS
# ============================================================

def create_agents():
    manager_agent = Agent(
        role      = "Project Manager",
        goal      = "Plan and coordinate business report creation.",
        backstory = "Experienced project manager who plans clearly.",
        llm              = llm,
        allow_delegation = False,
        verbose          = False
    )
    research_agent = Agent(
        role      = "Senior Research Specialist",
        goal      = "Organize and summarize research data.",
        backstory = "Expert researcher who structures findings clearly.",
        llm              = llm,
        allow_delegation = False,
        verbose          = False
    )
    analyst_agent = Agent(
        role      = "Business Analyst",
        goal      = "Extract insights and identify trends.",
        backstory = "Data-driven analyst who finds patterns.",
        llm              = llm,
        allow_delegation = False,
        verbose          = False
    )
    data_agent = Agent(
        role      = "Senior Data Analyst",
        goal      = "Analyze structured data and extract insights.",
        backstory = "Expert at finding patterns in business data.",
        llm              = llm,
        allow_delegation = False,
        verbose          = False
    )
    writer_agent = Agent(
        role      = "Professional Report Writer",
        goal      = "Write professional business reports.",
        backstory = "Expert writer who creates executive-ready reports.",
        llm              = llm,
        allow_delegation = False,
        verbose          = False
    )
    return manager_agent, research_agent, analyst_agent, data_agent, writer_agent

# ============================================================
# REPORT GENERATION
# ============================================================

def run_agent(agent, task, status_text, delay=20):
    status_text.write(f"⏳ {agent.role} working...")
    time.sleep(delay)
    mini_crew = Crew(
        agents  = [agent],
        tasks   = [task],
        process = Process.sequential,
        verbose = False
    )
    try:
        result = mini_crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Agent failed: {str(e)}"


def generate_report(topic, search_data, stats_data, progress, status):
    manager, research, analyst, data_ag, writer = create_agents()
    results = {}

    # Task 1 — Planning
    progress.progress(10)
    status.write("🧠 Manager planning report structure...")
    planning_task = Task(
        description     = f"""Plan report on: {topic}
        List 3 research questions. Be brief.""",
        expected_output = "3 research questions.",
        agent           = manager
    )
    results["planning"] = run_agent(
        manager, planning_task, status, delay=15
    )
    progress.progress(25)

    # Task 2 — Research
    status.write("🔍 Research agent organizing findings...")
    research_task = Task(
        description     = f"""Organize this research about {topic}:
        {search_data}
        Summarize into 5 key bullet points.""",
        expected_output = "5 bullet research summary.",
        agent           = research,
        context         = [planning_task]
    )
    results["research"] = run_agent(
        research, research_task, status, delay=20
    )
    progress.progress(45)

    # Task 3 — Data analysis (if CSV provided)
    if stats_data:
        status.write("📊 Data agent analyzing numbers...")
        data_task = Task(
            description     = f"""Analyze this business data:
            {stats_data}
            Give: 3 insights, 2 warnings, 2 opportunities,
            health score 1-10.""",
            expected_output = "Data insights with health score.",
            agent           = data_ag
        )
        results["data"] = run_agent(
            data_ag, data_task, status, delay=20
        )
        progress.progress(65)

    # Task 4 — Analysis
    status.write("💡 Analyst extracting insights...")
    context_text = results.get("research", "")
    if stats_data:
        context_text += "\n" + results.get("data", "")

    analyst_task = Task(
        description     = f"""Analyze findings about {topic}:
        {context_text}
        Give: 3 insights, 2 risks, 2 opportunities, SWOT.""",
        expected_output = "Analysis with SWOT.",
        agent           = analyst
    )
    results["analysis"] = run_agent(
        analyst, analyst_task, status, delay=20
    )
    progress.progress(80)

    # Task 5 — Report writing
    status.write("✍️ Writer creating final report...")
    full_context = "\n".join([
        results.get("research",  ""),
        results.get("data",      ""),
        results.get("analysis",  "")
    ])
    #
    writer_task = Task(
        description = f"""You are writing a report based on this exact data:

        ACTUAL DATA PROVIDED:
        {full_context}

        COLUMN NAMES IN DATA: {list(st.session_state.clean_df.columns) if st.session_state.clean_df is not None else 'No CSV'}

        YOUR RULES:
        - Read the column names carefully
        - Understand what type of data this is from the columns
        - Write report title based on actual data type
        - Every single claim must come from the numbers above
        - If you see temperature columns write weather report
        - If you see revenue columns write sales report
        - If you see population columns write demographic report
        - NEVER mention e-commerce, social media, online sales
          unless those exact words appear in the column names
        - If data has no clear business meaning say so honestly

        Topic context: {topic}

        Structure:
        1. Executive Summary (what this data actually shows)
        2. Key Findings (actual numbers from data)
        3. {'Data Analysis (actual column analysis),' if stats_data else ''}
        4. Patterns and Trends (from actual values)
        5. Recommendations (based on actual findings)
        6. Conclusion

        Maximum 450 words. Be honest about what the data shows.""",
        expected_output = "Report strictly based on actual provided data columns and values.",
        agent           = writer
    )
    
    results["report"] = run_agent(
        writer, writer_task, status, delay=20
    )
    progress.progress(100)

    return results

# ============================================================
# PDF EXPORT
# ============================================================

def export_report_pdf(report_text, topic):
    try:
        import tempfile

        doc  = fitz.open()
        page = doc.new_page()

        NAVY  = (0.10, 0.14, 0.49)
        BLACK = (0.13, 0.13, 0.13)
        GRAY  = (0.46, 0.46, 0.46)

        y = 50

        page.insert_text(
            (50, y),
            f"Business Report: {topic}",
            fontsize=18, color=NAVY
        )
        y += 30

        page.insert_text(
            (50, y),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=9, color=GRAY
        )
        y += 25

        page.draw_line((50, y), (545, y), color=NAVY, width=1)
        y += 20

        for line in report_text.split("\n"):
            if y > 750:
                page = doc.new_page()
                y    = 50

            line = line.strip()
            if not line:
                y += 8
                continue

            if any(h in line.lower() for h in [
                "executive", "findings", "recommendation",
                "conclusion", "overview", "risks", "swot",
                "analysis", "market"
            ]):
                page.insert_text(
                    (50, y), line,
                    fontsize=12, color=NAVY
                )
                y += 18
            else:
                max_chars = 90
                while len(line) > max_chars:
                    page.insert_text(
                        (50, y), line[:max_chars],
                        fontsize=10, color=BLACK
                    )
                    y    += 14
                    line  = line[max_chars:]
                    if y > 750:
                        page = doc.new_page()
                        y    = 50
                if line:
                    page.insert_text(
                        (50, y), line,
                        fontsize=10, color=BLACK
                    )
                    y += 14

        page.insert_text(
            (50, 800),
            "Generated by AI Business Report Generator",
            fontsize=8, color=GRAY
        )
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        doc.save(tmp.name)
        doc.close()

        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()

        try:
            os.unlink(tmp.name)
        except:
            pass
        return pdf_bytes

    except Exception as e:
        st.warning(f"PDF export error: {e}")
        return None

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp { background-color: #f8faff; }

    .title-box {
        background: linear-gradient(135deg, #1a237e, #3f51b5);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .title-box h1 { color: white; font-size: 28px; margin: 0; font-weight: 700; }
    .title-box p  { color: #c5cae9; margin: 6px 0 0 0; font-size: 14px; }

    .report-box {
        background: white;
        border: 1px solid #e8eaf6;
        border-left: 4px solid #3f51b5;
        border-radius: 12px;
        padding: 24px;
        color: #212121;
        font-size: 15px;
        line-height: 1.8;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .stat-card    { background: white; border: 1px solid #e8eaf6; border-radius: 10px; padding: 16px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .stat-number  { font-size: 28px; font-weight: 700; color: #3f51b5; }
    .stat-label   { font-size: 12px; color: #757575; margin-top: 4px; }

    .agent-step   { background: #e8eaf6; border-radius: 8px; padding: 10px 14px; margin: 4px 0; font-size: 13px; color: #1a237e; font-weight: 600; }
    .memory-card  { background: #f3f4ff; border: 1px solid #c5cae9; border-radius: 8px; padding: 12px; margin: 6px 0; font-size: 13px; color: #3f51b5; }

    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e8eaf6; }

    .stButton button {
        background: linear-gradient(135deg, #1a237e, #3f51b5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }

    div[data-testid="stTextInput"] input {
        background: white !important;
        border: 1.5px solid #c5cae9 !important;
        border-radius: 8px !important;
        color: #212121 !important;
        font-size: 15px !important;
    }

    .stProgress > div > div { background: linear-gradient(90deg, #1a237e, #3f51b5) !important; }
    hr { border-color: #e8eaf6 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "reports"           not in st.session_state: st.session_state.reports           = []
if "total_reports"     not in st.session_state: st.session_state.total_reports     = 0
if "current_report"    not in st.session_state: st.session_state.current_report    = None
if "clean_df"          not in st.session_state: st.session_state.clean_df          = None
if "stats_summary"     not in st.session_state: st.session_state.stats_summary     = None
if "eval_results"      not in st.session_state: st.session_state.eval_results      = None
if "pdf_text"          not in st.session_state: st.session_state.pdf_text          = None
if "show_agent_outputs" not in st.session_state: st.session_state.show_agent_outputs = False

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a237e,#3f51b5);
    border-radius:10px;padding:16px;margin-bottom:16px'>
        <h3 style='color:white;margin:0;font-size:18px'>📊 Report Generator</h3>
        <p style='color:#c5cae9;font-size:12px;margin:4px 0 0 0'>Multi-Agent AI System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    memories   = get_all_memories()
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.total_reports}</div>
            <div class='stat-label'>Reports Generated</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{len(memories)}</div>
            <div class='stat-label'>In Memory</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if memories:
        st.markdown(
            "<p style='color:#1a237e;font-weight:700;font-size:13px;margin-bottom:8px'>"
            "📚 Past Reports</p>",
            unsafe_allow_html=True
        )
        for mem in memories[-5:]:
            st.markdown(f"""
            <div class='memory-card'>
                📄 {mem['topic'][:35]}{'...' if len(mem['topic']) > 35 else ''}<br>
                <span style='color:#9fa8da;font-size:11px'>{mem['timestamp']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Session"):
        st.session_state.reports             = []
        st.session_state.total_reports       = 0
        st.session_state.current_report      = None
        st.session_state.clean_df            = None
        st.session_state.stats_summary       = None
        st.session_state.eval_results        = None
        st.session_state.pdf_text            = None
        st.session_state.show_agent_outputs  = False
        st.rerun()

    st.markdown(f"""
    <div style='background:#f3f4ff;border:1px solid #c5cae9;
    border-radius:8px;padding:10px;margin-top:12px'>
        <p style='color:#5c6bc0;font-size:11px;margin:0'>
        🤖 LLaMA 3.3 70B via Groq<br>
        🔍 DuckDuckGo Search<br>
        🧠 FAISS Memory<br>
        📊 Pandas ETL
        </p>
    </div>""", unsafe_allow_html=True)

# ============================================================
# MAIN AREA — Header
# ============================================================

st.markdown("""
<div class='title-box'>
    <h1>📊 AI Business Report Generator</h1>
    <p>Multi-agent system — Research + Analysis + Data + Writing in one click</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("### 🎯 Generate a Business Report")

col_topic, col_mode = st.columns([3, 1])

with col_topic:
    topic = st.text_input(
        "topic",
        placeholder      = "e.g. Analyze e-commerce market in Pakistan...",
        label_visibility = "collapsed"
    )

with col_mode:
    mode = st.selectbox(
        "Mode",
        ["🔍 Research Only", "📊 Data Only", "🚀 Combined"],
        label_visibility = "collapsed"
    )

# File uploads
col_csv, col_pdf = st.columns(2)

uploaded_csv = None
uploaded_pdf = None

with col_csv:
    if mode in ["📊 Data Only", "🚀 Combined"]:
        uploaded_csv = st.file_uploader(
            "📊 Upload CSV data file",
            type = ["csv"],
            help = "Upload business data for analysis"
        )
        if uploaded_csv:
            raw_df   = pd.read_csv(uploaded_csv)
            clean_df = run_etl(raw_df)
            stats    = generate_stats(clean_df)
            st.session_state.clean_df      = clean_df
            st.session_state.stats_summary = stats
            st.success(f"✅ CSV ready — {len(clean_df)} rows")
            st.info("💡 Tip: For best results upload business CSVs with clear column names like Revenue, Sales, Orders, Customers etc.")
            with st.expander("Preview data"):
                st.dataframe(clean_df.head(5))

with col_pdf:
    if mode == "🚀 Combined":
        uploaded_pdf = st.file_uploader(
            "📄 Upload PDF document",
            type = ["pdf"],
            help = "Upload research document or report"
        )
        if uploaded_pdf:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            pdf_text = extract_pdf_text(tmp_path)
            os.unlink(tmp_path)
            st.session_state.pdf_text = pdf_text
            st.success(f"✅ PDF ready — {len(pdf_text)} chars")

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    run_evaluation = st.checkbox(
        "🔍 Run evaluation + auto-improve report (Phase 8)",
        value = False,
        help  = "Agents will score and improve report quality automatically"
    )
    show_agent_steps = st.checkbox(
        "👁️ Show detailed agent steps",
        value = True
    )

# Generate button
generate_clicked = st.button("🚀 Generate Report", use_container_width=True)

# ============================================================
# REPORT GENERATION
# ============================================================

if generate_clicked:

    if not topic.strip() and mode == "🔍 Research Only":
        st.warning("⚠️ Please enter a topic first")

    elif mode == "📊 Data Only" and st.session_state.clean_df is None:
        st.warning("⚠️ Please upload a CSV file first")

    else:
        st.markdown("---")
        st.markdown("### ⚙️ Agent Pipeline Running")

        col_steps, col_progress = st.columns([1, 2])

        with col_steps:
            if show_agent_steps:
                st.markdown("""
                <div class='agent-step'>1️⃣ Manager Agent</div>
                <div class='agent-step'>2️⃣ Research Agent</div>
                <div class='agent-step'>3️⃣ Data Agent</div>
                <div class='agent-step'>4️⃣ Analyst Agent</div>
                <div class='agent-step'>5️⃣ Writer Agent</div>
                """, unsafe_allow_html=True)

                if run_evaluation:
                    st.markdown("""
                    <div class='agent-step'>6️⃣ Evaluation Agent</div>
                    <div class='agent-step'>7️⃣ Critic Agent</div>
                    <div class='agent-step'>8️⃣ Improver Agent</div>
                    """, unsafe_allow_html=True)

        with col_progress:
            progress = st.progress(0)
            status   = st.empty()
            status.write("🚀 Starting agents...")

            search_data = ""
            stats_data  = ""
            pdf_context = ""

            if mode in ["🔍 Research Only", "🚀 Combined"] and topic:
                status.write("🔍 Searching web...")
                search_data = web_search(f"{topic} market size trends 2024")

            if mode in ["📊 Data Only", "🚀 Combined"]:
                stats_data = st.session_state.stats_summary or ""

            if mode == "🚀 Combined":
                pdf_context = st.session_state.get("pdf_text", "")

            # Run main agents
            results = generate_report(
                topic       = topic or "Business Data Analysis",
                search_data = search_data + pdf_context,
                stats_data  = stats_data,
                progress    = progress,
                status      = status
            )

            report_text = results.get("report", "No report generated")

            # Phase 8 — Evaluation pipeline
            if run_evaluation:
                status.write("🔍 Running evaluation pipeline...")
                progress.progress(85)

                eval_agent = Agent(
                    role      = "Quality Evaluation Specialist",
                    goal      = "Evaluate report quality and score it.",
                    backstory = "Senior QA specialist with high standards.",
                    llm              = llm,
                    allow_delegation = False,
                    verbose          = False
                )

                eval_task = Task(
                    description = f"""Evaluate this report on {topic}:
                    {report_text[:1200]}

                    Score 1-10 each:
                    1. Completeness
                    2. Data Quality
                    3. Insight Depth
                    4. Professionalism
                    5. Actionability

                    Format exactly:
                    Completeness: X/10
                    Data Quality: X/10
                    Insight Depth: X/10
                    Professionalism: X/10
                    Actionability: X/10
                    Overall Score: X/10
                    Verdict: PASS or FAIL
                    Improvements:
                    1. improvement
                    2. improvement
                    3. improvement""",
                    expected_output  = "Scores, verdict and improvements.",
                    agent            = eval_agent
                )

                time.sleep(20)
                eval_crew = Crew(
                    agents  = [eval_agent],
                    tasks   = [eval_task],
                    process = Process.sequential,
                    verbose = False
                )

                try:
                    eval_result  = str(eval_crew.kickoff())
                    st.session_state.eval_results = eval_result

                    if "FAIL" in eval_result.upper():
                        status.write("⚠️ Report failed — improving...")
                        improver = Agent(
                            role      = "Report Improver",
                            goal      = "Improve report based on feedback.",
                            backstory = "Expert at improving reports.",
                            llm              = llm,
                            allow_delegation = False,
                            verbose          = False
                        )
                        improve_task = Task(
                            description     = f"""Improve this report:
                            {report_text[:1200]}
                            Based on feedback:
                            {eval_result[:500]}
                            Fix all issues. Max 450 words.""",
                            expected_output = "Improved 450 word report.",
                            agent           = improver
                        )
                        time.sleep(20)
                        improve_crew = Crew(
                            agents  = [improver],
                            tasks   = [improve_task],
                            process = Process.sequential,
                            verbose = False
                        )
                        report_text = str(improve_crew.kickoff())
                        status.write("✅ Report improved!")
                    else:
                        status.write("✅ Report passed evaluation!")

                except Exception as e:
                    st.warning(f"Evaluation failed: {e}")

            progress.progress(100)
            status.write("✅ All done!")

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"outputs/report_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"TOPIC    : {topic}\n")
            f.write(f"MODE     : {mode}\n")
            f.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(report_text)

        save_to_memory(topic or "Data Analysis", report_text)

        st.session_state.current_report = report_text
        st.session_state.total_reports += 1
        st.session_state.reports.append({
            "topic"    : topic,
            "report"   : report_text,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        st.rerun()

# ============================================================
# DISPLAY REPORT
# ============================================================

if st.session_state.current_report:
    st.markdown("---")
    st.markdown("### 📄 Generated Report")

    # Sources used badges
    if st.session_state.reports:
        last_topic   = st.session_state.reports[-1].get("topic", "")
        sources_used = []
        if last_topic:
            sources_used.append("🔍 Web Research")
        if st.session_state.stats_summary:
            sources_used.append("📊 CSV Data")
        if st.session_state.get("pdf_text"):
            sources_used.append("📄 PDF Document")

        if sources_used:
            st.markdown(
                " &nbsp; ".join([
                    f"<span style='background:#e8eaf6;color:#1a237e;"
                    f"padding:4px 10px;border-radius:12px;"
                    f"font-size:12px;font-weight:600'>{s}</span>"
                    for s in sources_used
                ]),
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # Evaluation results
    if st.session_state.get("eval_results"):
        with st.expander("📊 Evaluation Scores (Phase 8)"):
            eval_text = st.session_state.eval_results
            if "PASS" in eval_text.upper():
                st.markdown("""
                <div style='background:#e8f5e9;border:1px solid #a5d6a7;
                border-radius:8px;padding:10px;margin-bottom:10px'>
                    <span style='color:#2e7d32;font-weight:700;font-size:16px'>
                    ✅ PASS — Report meets quality standards</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#fff8e1;border:1px solid #ffe082;
                border-radius:8px;padding:10px;margin-bottom:10px'>
                    <span style='color:#f57f17;font-weight:700;font-size:16px'>
                    ⚠️ IMPROVED — Report was auto-enhanced</span>
                </div>""", unsafe_allow_html=True)
            st.text(eval_text)

    # CSV Charts
    if st.session_state.clean_df is not None:
        with st.expander("📈 Data Charts (Phase 9)"):
            df       = st.session_state.clean_df
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()

            if num_cols and str_cols:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown(f"**{num_cols[0]} by {str_cols[0]}**")
                    st.bar_chart(df.set_index(str_cols[0])[num_cols[0]])
                if len(num_cols) > 1:
                    with col_c2:
                        st.markdown("**Multi-metric Trend**")
                        st.line_chart(df.set_index(str_cols[0])[num_cols[:2]])

    # Report content
    st.markdown(f"""
    <div class='report-box'>
        {st.session_state.current_report.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # Download buttons
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.download_button(
            label               = "⬇️ Download TXT",
            data                = st.session_state.current_report,
            file_name           = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime                = "text/plain",
            use_container_width = True
        )

    with col_dl2:
        pdf_bytes = export_report_pdf(
            st.session_state.current_report,
            st.session_state.reports[-1].get("topic", "Report") if st.session_state.reports else "Report"
        )
        if pdf_bytes:
            st.download_button(
                label               = "📥 Download PDF",
                data                = pdf_bytes,
                file_name           = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime                = "application/pdf",
                use_container_width = True
            )

    # Agent outputs viewer
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("🗂️ View Agent Outputs", use_container_width=True):
            st.session_state.show_agent_outputs = not st.session_state.show_agent_outputs

    if st.session_state.show_agent_outputs:
        with st.expander("🔍 Agent Outputs", expanded=True):
            if st.session_state.reports:
                last = st.session_state.reports[-1]
                st.markdown(f"**Topic:** {last['topic'] if last['topic'] else 'Data Analysis'}")
                st.markdown(f"**Time:** {last['timestamp']}")
                st.markdown(f"**Mode:** {mode}")
                st.markdown(f"**Reports this session:** {st.session_state.total_reports}")