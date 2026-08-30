import streamlit as st

from agent_backend_seventhnb import (
    ask_agent,
    employees
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NexaTech Employee Assistant",
    page_icon="🏢",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("🏢 NexaTech Employee Assistant")

st.caption(
    "AI-powered attendance, leave and company "
    "policy assistant"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Employee")

employee_ids = employees[
    "employee_id"
].tolist()

employee_id = st.sidebar.selectbox(
    "Select employee",
    employee_ids
)


# Find selected employee

employee_data = employees[
    employees["employee_id"] == employee_id
].iloc[0]


st.sidebar.write(
    f"**Name:** {employee_data['name']}"
)

st.sidebar.write(
    f"**Department:** "
    f"{employee_data['department']}"
)

st.sidebar.write(
    f"**Role:** {employee_data['role']}"
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about attendance, leave, working hours or company policy..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Show user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Generate response

    with st.chat_message("assistant"):

        with st.spinner(
            "Checking company data..."
        ):

            try:

                answer = ask_agent(
                    question=question,
                    employee_id=employee_id
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "Sorry, something went wrong "
                    "while processing your request."
                )

                st.error(
                    f"{answer}\n\nError: {e}"
                )


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )