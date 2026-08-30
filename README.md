# NexaTech Employee Assistant

An AI-powered conversational assistant for managing employee attendance, leave, working hours, and company policy queries. Built with LangChain, ChromaDB, and Streamlit.

## Overview

The NexaTech Employee Assistant is an intelligent agent that combines **Retrieval-Augmented Generation (RAG)** with structured data analysis to provide accurate, policy-aware responses about employee attendance and company regulations.

### Key Features

- 🤖 **AI-Powered Agent**: ReAct-based agent using Groq's LLM for intelligent decision-making
- 📋 **Policy Retrieval**: Vector-based semantic search over company policy documents
- 📊 **Attendance Analytics**: Real-time analysis of attendance records and working hours
- ✅ **Compliance Checking**: Automatic verification against company policies
- 👥 **Employee Information**: Quick access to employee details and entitlements
- 💬 **Conversational Interface**: Natural language chat interface via Streamlit

## Technologies Used

- **LLM**: Groq (OpenAI GPT-OSS 120B) with reasoning capabilities
- **RAG Framework**: LangChain Core + Text Splitters
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **Agent Framework**: LangGraph (ReAct agent)
- **UI**: Streamlit
- **Data**: Pandas + CSV
- **Environment**: Python 3.14+ with UV package manager

## Project Structure

```
attendance-rag/
├── app.py                          # Streamlit UI application
├── agent_backend_seventhnb.py      # Agent logic, tools, and RAG setup
├── pyproject.toml                  # Project dependencies and configuration
├── .env                            # Environment variables (GROQ_API_KEY)
├── data/
│   ├── structured/
│   │   ├── employees.csv           # Employee master data
│   │   ├── attendance.csv          # Daily attendance records
│   │   ├── leave.csv               # Leave request records
│   │   └── holidays.csv            # Company holidays
│   └── knowledge_base/
│       ├── attendance_policy.md
│       ├── leave_policy.md
│       ├── overtime_policy.md
│       ├── remote_work_policy.md
│       ├── sick_leave_policy.md
│       └── working_hours_policy.md
├── notebooks/                      # Development notebooks
│   ├── 01_rag.ipynb
│   ├── 02_structured_data.ipynb
│   ├── 03_agent_tools.ipynb
│   ├── 04_agent.ipynb
│   ├── 05_agent_with_rag.ipynb
│   ├── 06_compliance_tools.ipynb
│   └── 07_final_agent.ipynb
└── src/
    └── attendance_rag/
        └── __init__.py
```

## Data Note

⚠️ **All data in this project is synthetic and AI-generated** for demonstration purposes:
- Employee records, attendance logs, leave data, and policies are fictional
- Suitable for development, testing, and proof-of-concept demonstrations
- Not intended for production use with real employee data
- To use with actual data, replace CSVs in `data/structured/` with your real data

## Installation

### Prerequisites
- Python 3.14+
- UV package manager
- Groq API key

### Setup Steps

1. **Clone and navigate to project**
   ```bash
   cd "New folder/Attendance RAG"
   ```

2. **Create environment file**
   ```bash
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```
   Get your API key from [console.groq.com](https://console.groq.com)

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Activate virtual environment**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

## Running the Application

### Launch Streamlit UI
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the Chat Interface

1. **Select an employee** from the sidebar (e.g., E0001)
2. **Type your question** in the chat input
3. **Get AI-powered answers** with policy references and data analysis

## Screenshots

See the `screenshots/` folder for examples of the three queries in action:
- Attendance query: "How many days did E0001 work from home in August 2026?"
- Policy query: "What is the company's policy regarding working from home?"
- Compliance query: "Did E0001 comply with the office attendance requirement in August 2026?"

## How It Works

### Architecture

```
User Query
    ↓
Streamlit UI (app.py)
    ↓
ReAct Agent (LangGraph)
    ├─→ Tool 1: search_company_policy (RAG via ChromaDB)
    ├─→ Tool 2: get_employee (DataFrame lookup)
    ├─→ Tool 3: get_attendance_summary (Date range filter)
    ├─→ Tool 4: calculate_working_hours (Time calculation)
    ├─→ Tool 5: get_leave_balance (Entitlement calculation)
    ├─→ Tool 6: find_missing_attendance (Data validation)
    ├─→ Tool 7: check_remote_work_compliance (Policy check)
    └─→ Tool 8: check_working_hours_compliance (Hour verification)
    ↓
Groq LLM (Reasoning + Tool Selection)
    ↓
Final Answer
```

### Tools Available

| Tool | Purpose |
|------|---------|
| `search_company_policy` | Semantic search over policy documents via RAG |
| `get_employee` | Retrieve employee info (name, dept, role, hours, location) |
| `get_attendance_summary` | Count office/remote/sick/leave days in date range |
| `calculate_working_hours` | Sum recorded working hours from check-in/out times |
| `get_leave_balance` | Calculate vacation used vs entitlement |
| `find_missing_attendance` | Identify missing check-in/check-out records |
| `check_remote_work_compliance` | Verify minimum 3 office days/week requirement |
| `check_working_hours_compliance` | Compare recorded vs expected hours |

### RAG Pipeline

1. **Load**: Policy documents (6 markdown files) from `data/knowledge_base/`
2. **Split**: Recursive character splitting (700 chars, 150 overlap)
3. **Embed**: HuggingFace sentence transformers
4. **Store**: ChromaDB vector collection
5. **Retrieve**: Top-4 semantic matches for policy queries

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your key from [Groq Console](https://console.groq.com)

## Development

### Notebooks

The `notebooks/` folder contains step-by-step development:
- `01_rag.ipynb` - RAG setup with policy documents
- `02_structured_data.ipynb` - Data loading and exploration
- `03_agent_tools.ipynb` - Tool definitions
- `04_agent.ipynb` - Basic agent setup
- `05_agent_with_rag.ipynb` - Agent + RAG integration
- `06_compliance_tools.ipynb` - Compliance checking logic
- `07_final_agent.ipynb` - Final agent with all tools

### Data Format

**employees.csv**
```
employee_id, name, department, role, employment_type, weekly_hours, vacation_days, office_location
```

**attendance.csv**
```
employee_id, date, status, location, check_in, check_out
```

**leave.csv**
```
employee_id, type, start_date, end_date, days, status
```

**holidays.csv**
```
date, holiday_name
```

## Troubleshooting

### "No relevant company policy was found"
- Policy documents may not contain the topic
- Try rephrasing the question
- Check that markdown files exist in `data/knowledge_base/`

### "No employee found with ID X"
- Verify the employee ID exists in `employees.csv`
- Check the exact spelling/format

### LLM timeout or error
- Verify `GROQ_API_KEY` is set in `.env`
- Check Groq API status at [console.groq.com](https://console.groq.com)
- Ensure internet connection is active

### No attendance records found
- Verify attendance data in CSV matches the query date range
- Check employee IDs are consistent across CSVs

## Future Enhancements

- [ ] Export compliance reports to PDF
- [ ] Historical analytics and trends
- [ ] Multi-language support
- [ ] Integration with HR systems
- [ ] Automated compliance alerts
- [ ] Performance dashboards

## License

This project is for internal NexaTech use.

## Questions?

For issues or questions, refer to the development notebooks or the inline code documentation in `agent_backend_seventhnb.py`.
