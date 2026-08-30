from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.tools import tool

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

STRUCTURED_DIR = DATA_DIR / "structured"

KB_DIR = DATA_DIR / "knowledge_base"


# ============================================================
# 3. LOAD STRUCTURED DATA
# ============================================================

employees = pd.read_csv(
    STRUCTURED_DIR / "employees.csv"
)

attendance = pd.read_csv(
    STRUCTURED_DIR / "attendance.csv"
)

leave = pd.read_csv(
    STRUCTURED_DIR / "leave.csv"
)

holidays = pd.read_csv(
    STRUCTURED_DIR / "holidays.csv"
)


attendance["date"] = pd.to_datetime(
    attendance["date"]
)

leave["start_date"] = pd.to_datetime(
    leave["start_date"]
)

leave["end_date"] = pd.to_datetime(
    leave["end_date"]
)

holidays["date"] = pd.to_datetime(
    holidays["date"]
)


# ============================================================
# 4. LOAD KNOWLEDGE BASE
# ============================================================

documents = []

for file in KB_DIR.glob("*.md"):

    text = file.read_text(
        encoding="utf-8"
    )

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": file.name
            }
        )
    )


# ============================================================
# 5. SPLIT DOCUMENTS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ".",
        " "
    ]
)

chunks = splitter.split_documents(
    documents
)


# ============================================================
# 6. EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 7. VECTOR DATABASE
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="nexatech_streamlit_policies"
)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# 8. RAG TOOL
# ============================================================

@tool
def search_company_policy(
    question: str
) -> str:
    """
    Search company policy documents.

    Use this for questions about company rules,
    attendance requirements, remote work,
    working hours, overtime, leave and holidays.

    Do not use this for an employee's actual
    attendance records.
    """

    retrieved_docs = retriever.invoke(
        question
    )

    if not retrieved_docs:

        return (
            "No relevant company policy was found."
        )

    results = []

    for doc in retrieved_docs:

        results.append(
            f"Source: {doc.metadata['source']}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(
        results
    )


# ============================================================
# 9. EMPLOYEE TOOL
# ============================================================

@tool
def get_employee(
    employee_id: str
) -> str:
    """
    Get employee information.
    """

    result = employees[
        employees["employee_id"] == employee_id
    ]

    if result.empty:

        return (
            f"No employee found with ID "
            f"{employee_id}."
        )

    employee = result.iloc[0]

    return (
        f"Employee ID: {employee['employee_id']}\n"
        f"Name: {employee['name']}\n"
        f"Department: {employee['department']}\n"
        f"Role: {employee['role']}\n"
        f"Employment type: {employee['employment_type']}\n"
        f"Weekly hours: {employee['weekly_hours']}\n"
        f"Vacation entitlement: "
        f"{employee['vacation_days']} days\n"
        f"Office location: "
        f"{employee['office_location']}"
    )


# ============================================================
# 10. ATTENDANCE TOOL
# ============================================================

@tool
def get_attendance_summary(
    employee_id: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Get attendance statistics for an employee during a date range.
    
    Returns counts of:
    - Office days (on-site work)
    - Home-office days (work from home / remote work)
    - Business trip days
    - Sick days
    - Leave days
    - Missing attendance records
    
    Use this for questions about how many days someone worked
    from home, office, or on business trips.
    """

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    records = attendance[
        (attendance["employee_id"] == employee_id)
        &
        (attendance["date"] >= start)
        &
        (attendance["date"] <= end)
    ]

    if records.empty:

        return (
            f"No attendance records found "
            f"for {employee_id}."
        )

    office_days = (
        records["location"] == "office"
    ).sum()

    home_days = (
        records["location"] == "home"
    ).sum()

    business_trip_days = (
        records["location"] == "business_trip"
    ).sum()

    sick_days = (
        records["status"] == "sick"
    ).sum()

    leave_days = (
        records["status"] == "leave"
    ).sum()

    missing_records = records[
        records["status"].isin(
            [
                "missing",
                "missing_checkout"
            ]
        )
    ].shape[0]

    return (
        f"Employee: {employee_id}\n"
        f"Period: {start_date} to {end_date}\n"
        f"Office days: {office_days}\n"
        f"Home-office days: {home_days}\n"
        f"Business-trip days: "
        f"{business_trip_days}\n"
        f"Sick days: {sick_days}\n"
        f"Leave days: {leave_days}\n"
        f"Missing attendance records: "
        f"{missing_records}"
    )


# ============================================================
# 11. WORKING HOURS TOOL
# ============================================================

@tool
def calculate_working_hours(
    employee_id: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Calculate total recorded working hours.
    """

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    records = attendance[
        (attendance["employee_id"] == employee_id)
        &
        (attendance["date"] >= start)
        &
        (attendance["date"] <= end)
    ].copy()

    records = records[
        records["check_in"].notna()
        &
        records["check_out"].notna()
        &
        (records["check_in"] != "")
        &
        (records["check_out"] != "")
    ]

    if records.empty:

        return (
            f"No complete attendance records "
            f"found for {employee_id}."
        )

    records["check_in_time"] = pd.to_datetime(
        records["check_in"],
        format="%H:%M"
    )

    records["check_out_time"] = pd.to_datetime(
        records["check_out"],
        format="%H:%M"
    )

    records["hours"] = (
        records["check_out_time"]
        -
        records["check_in_time"]
    ).dt.total_seconds() / 3600

    total_hours = records["hours"].sum()

    return (
        f"Employee: {employee_id}\n"
        f"Period: {start_date} to {end_date}\n"
        f"Total recorded working hours: "
        f"{total_hours:.2f}"
    )


# ============================================================
# 12. LEAVE BALANCE TOOL
# ============================================================

@tool
def get_leave_balance(
    employee_id: str
) -> str:
    """
    Get vacation entitlement and remaining leave.
    """

    employee_result = employees[
        employees["employee_id"] == employee_id
    ]

    if employee_result.empty:

        return (
            f"No employee found with ID "
            f"{employee_id}."
        )

    employee = employee_result.iloc[0]

    entitlement = int(
        employee["vacation_days"]
    )

    approved_vacation = leave[
        (leave["employee_id"] == employee_id)
        &
        (leave["type"] == "vacation")
        &
        (leave["status"] == "approved")
    ]

    used = int(
        approved_vacation["days"].sum()
    )

    remaining = entitlement - used

    return (
        f"Employee: {employee_id}\n"
        f"Vacation entitlement: "
        f"{entitlement} days\n"
        f"Approved vacation used: "
        f"{used} days\n"
        f"Remaining vacation: "
        f"{remaining} days"
    )


# ============================================================
# 13. MISSING ATTENDANCE TOOL
# ============================================================

@tool
def find_missing_attendance(
    employee_id: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Find missing attendance records.
    """

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    records = attendance[
        (attendance["employee_id"] == employee_id)
        &
        (attendance["date"] >= start)
        &
        (attendance["date"] <= end)
    ]

    missing = records[
        records["status"].isin(
            [
                "missing",
                "missing_checkout"
            ]
        )
    ]

    if missing.empty:

        return (
            f"No missing attendance records "
            f"found for {employee_id}."
        )

    results = []

    for _, row in missing.iterrows():

        results.append(
            f"{row['date'].date()} - "
            f"{row['status']}"
        )

    return (
        f"Missing attendance for "
        f"{employee_id}:\n"
        +
        "\n".join(results)
    )


# ============================================================
# 14. POLICY CONFIGURATION
# ============================================================

POLICY_CONFIG = {
    "minimum_office_days_per_week": 3
}


# ============================================================
# 15. WEEKLY ATTENDANCE CALCULATION
# ============================================================

def calculate_weekly_attendance(
    employee_id,
    start_date,
    end_date
):

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    records = attendance[
        (attendance["employee_id"] == employee_id)
        &
        (attendance["date"] >= start)
        &
        (attendance["date"] <= end)
    ].copy()

    if records.empty:

        return pd.DataFrame()

    records["week"] = (
        records["date"]
        .dt.to_period("W-MON")
        .astype(str)
    )

    weekly = (
        records
        .groupby("week")
        .agg(
            office_days=(
                "location",
                lambda x:
                    (x == "office").sum()
            ),
            home_days=(
                "location",
                lambda x:
                    (x == "home").sum()
            )
        )
        .reset_index()
    )

    return weekly


# ============================================================
# 16. REMOTE WORK COMPLIANCE
# ============================================================

@tool
def check_remote_work_compliance(
    employee_id: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Determine whether an employee met the company's
    weekly office attendance requirement.
    """

    weekly = calculate_weekly_attendance(
        employee_id,
        start_date,
        end_date
    )

    if weekly.empty:

        return (
            f"No attendance data found "
            f"for {employee_id}."
        )

    required = POLICY_CONFIG[
        "minimum_office_days_per_week"
    ]

    weekly["compliant"] = (
        weekly["office_days"]
        >= required
    )

    failed = weekly[
        ~weekly["compliant"]
    ]

    overall = bool(
        weekly["compliant"].all()
    )

    status = (
        "COMPLIANT"
        if overall
        else "NOT COMPLIANT"
    )

    result = (
        f"Employee: {employee_id}\n"
        f"Period: {start_date} to {end_date}\n"
        f"Required office days per week: "
        f"{required}\n"
        f"Weeks checked: "
        f"{len(weekly)}\n"
        f"Compliant weeks: "
        f"{weekly['compliant'].sum()}\n"
        f"Non-compliant weeks: "
        f"{(~weekly['compliant']).sum()}\n"
        f"Overall status: {status}"
    )

    if not failed.empty:

        result += (
            "\n\nWeeks below requirement:\n"
        )

        for _, row in failed.iterrows():

            result += (
                f"- {row['week']}: "
                f"{row['office_days']} "
                f"office days\n"
            )

    return result


# ============================================================
# 17. WORKING HOURS COMPLIANCE
# ============================================================

@tool
def check_working_hours_compliance(
    employee_id: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Compare recorded working hours against
    contracted working hours.
    """

    employee_result = employees[
        employees["employee_id"] == employee_id
    ]

    if employee_result.empty:

        return (
            f"No employee found with ID "
            f"{employee_id}."
        )

    employee = employee_result.iloc[0]

    weekly_hours = float(
        employee["weekly_hours"]
    )

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    records = attendance[
        (attendance["employee_id"] == employee_id)
        &
        (attendance["date"] >= start)
        &
        (attendance["date"] <= end)
    ].copy()

    records = records[
        records["check_in"].notna()
        &
        records["check_out"].notna()
        &
        (records["check_in"] != "")
        &
        (records["check_out"] != "")
    ]

    if records.empty:

        return (
            "No complete attendance records "
            "were found."
        )

    records["check_in_time"] = pd.to_datetime(
        records["check_in"],
        format="%H:%M"
    )

    records["check_out_time"] = pd.to_datetime(
        records["check_out"],
        format="%H:%M"
    )

    records["hours"] = (
        records["check_out_time"]
        -
        records["check_in_time"]
    ).dt.total_seconds() / 3600

    recorded_hours = records["hours"].sum()

    days = (
        end - start
    ).days + 1

    expected_hours = (
        weekly_hours * days / 7
    )

    status = (
        "COMPLIANT"
        if recorded_hours >= expected_hours
        else "BELOW EXPECTED HOURS"
    )

    return (
        f"Employee: {employee_id}\n"
        f"Period: {start_date} to {end_date}\n"
        f"Contracted weekly hours: "
        f"{weekly_hours:.2f}\n"
        f"Recorded hours: "
        f"{recorded_hours:.2f}\n"
        f"Expected hours: "
        f"{expected_hours:.2f}\n"
        f"Status: {status}"
    )


# ============================================================
# 18. CREATE AGENT
# ============================================================

tools = [
    search_company_policy,
    get_employee,
    get_attendance_summary,
    calculate_working_hours,
    find_missing_attendance,
    get_leave_balance,
    check_remote_work_compliance,
    check_working_hours_compliance
]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    reasoning_format="parsed"
)


agent = create_react_agent(
    model=llm,
    tools=tools
)


# ============================================================
# 19. FUNCTION USED BY STREAMLIT
# ============================================================

def ask_agent(
    question: str,
    employee_id: str
) -> str:

    employee_context = f"""
The currently logged-in employee is:

Employee ID: {employee_id}

You MUST use this employee ID when answering
questions about attendance, leave, working hours,
or employee-specific information.

Do not access another employee's records.
"""

    full_question = (
        employee_context
        +
        "\n\nUser question:\n"
        +
        question
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": full_question
                }
            ]
        }
    )

    return response[
        "messages"
    ][-1].content