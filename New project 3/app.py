import csv
import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
import pandas as pd
import streamlit as st


# Configuration
DEFAULT_MODEL = "gemini-2.5-flash"
LOG_FILE = Path("support_logs.csv")
EXPECTED_FIELDS = (
    "issues",
    "primary_category",
    "priority",
    "confidence",
    "suggested_action",
    "workflow",
    "response",
    "reasoning",
)
VALID_CATEGORIES = {"billing", "delivery", "complaint", "account", "other"}
VALID_PRIORITIES = {"Low", "Medium", "High"}
SAMPLE_QUERIES = [
    "I want a refund for my order",
    "My order is delayed and I was charged twice",
    "I cannot log in to my account",
]
PRIORITY_STYLES = {
    "High": ("#f87171", "#450a0a"),
    "Medium": ("#facc15", "#422006"),
    "Low": ("#4ade80", "#052e16"),
}


class SupportResponseParseError(ValueError):
    """Raised when Gemini output cannot be converted into the expected schema."""


# Page setup
def configure_page():
    st.set_page_config(
        page_title="AI Customer Support Copilot",
        page_icon=":speech_balloon:",
        layout="wide",
    )


def apply_custom_styles():
    st.markdown(
        """
        <style>
        .chat-bubble {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.35);
            margin-bottom: 1rem;
        }
        .chat-label {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }
        .priority-pill {
            border-radius: 999px;
            display: inline-block;
            font-weight: 700;
            padding: 0.2rem 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Gemini integration
@st.cache_resource
def get_gemini_client():
    return genai.Client()


def build_prompt(customer_query):
    return (
        "You are an AI customer support assistant for an e-commerce company.\n\n"
        "Analyze the customer query and return ONLY a valid JSON object.\n\n"
        "The JSON must contain:\n"
        "- issues (list of detected issues if multiple, else single item)\n"
        "- primary_category (billing, delivery, complaint, account, other)\n"
        "- priority (Low, Medium, High)\n"
        "- confidence (0 to 1 score based on clarity of intent)\n"
        "- suggested_action\n"
        "- workflow (step-by-step internal actions as a list)\n"
        "- response (professional reply)\n"
        "- reasoning (short explanation of classification)\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON\n"
        "- No markdown, no backticks, no explanation\n"
        "- Use double quotes\n"
        "- Keep response concise\n\n"
        "Examples:\n\n"
        "Input: I want a refund for my order\n"
        "Output:\n"
        "{\n"
        '  "issues": ["refund request"],\n'
        '  "primary_category": "billing",\n'
        '  "priority": "High",\n'
        '  "confidence": 0.95,\n'
        '  "suggested_action": "Initiate refund",\n'
        '  "workflow": ["Verify order", "Check payment", "Process refund"],\n'
        '  "response": "We apologize for the inconvenience. Your refund request has been initiated.",\n'
        '  "reasoning": "User explicitly asked for refund"\n'
        "}\n\n"
        "Input: My order is delayed and I was charged twice\n"
        "Output:\n"
        "{\n"
        '  "issues": ["delivery delay", "double charge"],\n'
        '  "primary_category": "billing",\n'
        '  "priority": "High",\n'
        '  "confidence": 0.9,\n'
        '  "suggested_action": "Investigate payment and shipment",\n'
        '  "workflow": ["Check payment logs", "Verify shipment status", "Escalate if needed"],\n'
        '  "response": "We are sorry for the inconvenience. We are checking both your payment and delivery status.",\n'
        '  "reasoning": "Multiple issues detected: billing and delivery"\n'
        "}\n\n"
        "Customer Query:\n"
        f"{customer_query}"
    )


def call_gemini(customer_query):
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = get_gemini_client()

    response = client.models.generate_content(
        model=model,
        contents=build_prompt(customer_query),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    return response.text


def generate_with_retry(customer_query, retries=1):
    last_error = None

    for attempt in range(retries + 1):
        try:
            return call_gemini(customer_query)
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1)

    raise last_error


# JSON extraction and validation
def strip_markdown_formatting(response_text):
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def extract_json_text(response_text):
    text = strip_markdown_formatting(response_text)
    decoder = json.JSONDecoder()
    start_index = text.find("{")

    if start_index == -1:
        raise SupportResponseParseError("No JSON object was found in the model response.")

    try:
        _, relative_end_index = decoder.raw_decode(text[start_index:])
    except json.JSONDecodeError as error:
        raise SupportResponseParseError("The extracted content is not valid JSON.") from error

    end_index = start_index + relative_end_index
    return text[start_index:end_index]


def parse_json_text(json_text):
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as error:
        raise SupportResponseParseError("The JSON object could not be parsed.") from error


def validate_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise SupportResponseParseError(f"The {field} field must be a non-empty string.")


def validate_string_list(value, field):
    if not isinstance(value, list) or not value:
        raise SupportResponseParseError(f"The {field} field must be a non-empty list.")

    for item in value:
        validate_string(item, field)


def validate_support_response(parsed_response):
    if not isinstance(parsed_response, dict):
        raise SupportResponseParseError("The response must be a JSON object.")

    if set(parsed_response) != set(EXPECTED_FIELDS):
        raise SupportResponseParseError(
            "The JSON response must contain exactly: "
            f"{', '.join(EXPECTED_FIELDS)}."
        )

    validate_string_list(parsed_response["issues"], "issues")
    validate_string_list(parsed_response["workflow"], "workflow")

    if parsed_response["primary_category"] not in VALID_CATEGORIES:
        raise SupportResponseParseError("The primary_category field contains an unsupported value.")

    if parsed_response["priority"] not in VALID_PRIORITIES:
        raise SupportResponseParseError("The priority field contains an unsupported value.")

    confidence = parsed_response["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise SupportResponseParseError("The confidence field must be a number.")

    if confidence < 0 or confidence > 1:
        raise SupportResponseParseError("The confidence field must be between 0 and 1.")

    for field in ("suggested_action", "response", "reasoning"):
        validate_string(parsed_response[field], field)

    parsed_response["confidence"] = float(confidence)
    return parsed_response


def parse_support_response(response_text):
    if not response_text or not response_text.strip():
        raise SupportResponseParseError("The model returned an empty response.")

    json_text = extract_json_text(response_text)
    parsed_response = parse_json_text(json_text)
    return validate_support_response(parsed_response)


# Logging and analytics
def log_interaction(customer_query, support_response):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": customer_query,
        "issues": "; ".join(support_response["issues"]),
        "category": support_response["primary_category"],
        "priority": support_response["priority"],
        "confidence": support_response["confidence"],
        "suggested_action": support_response["suggested_action"],
        "response": support_response["response"],
    }

    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_logs():
    if not LOG_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(LOG_FILE)


# UI helpers
def render_header():
    st.title("AI Customer Support Copilot")
    st.caption("Classify support tickets, recommend actions, and draft concise replies.")


def render_sample_buttons():
    st.write("Sample queries")
    columns = st.columns(len(SAMPLE_QUERIES))

    for index, sample_query in enumerate(SAMPLE_QUERIES):
        if columns[index].button(sample_query, use_container_width=True):
            st.session_state.customer_query = sample_query


def render_query_form():
    with st.container(border=True):
        st.subheader("Customer Query")
        render_sample_buttons()
        customer_query = st.text_area(
            "Enter the customer's message",
            key="customer_query",
            placeholder="Example: My order arrived damaged. Can I get a replacement?",
            height=160,
        )
        submitted = st.button(
            "Generate Response",
            type="primary",
            use_container_width=True,
        )

    return customer_query, submitted


def render_priority_badge(priority):
    text_color, background_color = PRIORITY_STYLES[priority]
    st.markdown(
        f"""
        <span class="priority-pill" style="color:{text_color}; background:{background_color};">
            {priority}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_chat_response(customer_query, support_response):
    st.markdown("### Generated Response")
    safe_customer_query = html.escape(customer_query)

    st.markdown(
        f"""
        <div class="chat-bubble">
            <div class="chat-label">Customer</div>
            <div>{safe_customer_query}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.caption("Support Copilot")
        st.write(support_response["response"])

        metric_columns = st.columns(3)
        metric_columns[0].metric("Category", support_response["primary_category"])
        metric_columns[1].metric("Confidence", f"{support_response['confidence']:.2f}")
        with metric_columns[2]:
            st.write("Priority")
            render_priority_badge(support_response["priority"])

        st.write(f"**Suggested Action:** {support_response['suggested_action']}")
        st.write("**Issues:** " + ", ".join(support_response["issues"]))
        st.write("**Workflow:**")
        for step_number, step in enumerate(support_response["workflow"], start=1):
            st.write(f"{step_number}. {step}")

        if st.toggle("Show reasoning"):
            st.info(support_response["reasoning"])

def render_parse_error(error, raw_response):
    st.error(
        "Gemini returned a response that could not be converted into the expected format. "
        "Please try again with a clearer customer query."
    )
    st.caption(f"Details: {error}")
    with st.expander("Show raw Gemini response"):
        st.code(raw_response or "No response text returned.", language="text")


def render_response(customer_query, submitted):
    if not submitted:
        st.info("Enter a customer query or choose a sample, then click Generate Response.")
        return

    cleaned_query = customer_query.strip()
    if not cleaned_query:
        st.warning("Please enter a customer query before generating a response.")
        return

    if len(cleaned_query) < 5:
        st.warning("Please enter a more complete customer query.")
        return

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Add it to your environment and try again.")
        return

    try:
        with st.spinner("Analyzing customer query..."):
            raw_response = generate_with_retry(cleaned_query, retries=1)
    except Exception as error:
        st.error(
            "Gemini API request failed after retrying once. "
            "Please check your API key, quota, network connection, and model name."
        )
        st.caption(f"Details: {error}")
        return

    try:
        support_response = parse_support_response(raw_response)
    except SupportResponseParseError as error:
        render_parse_error(error, raw_response)
        return

    log_interaction(cleaned_query, support_response)
    render_chat_response(cleaned_query, support_response)

def render_analytics_dashboard():
    logs = load_logs()
    st.subheader("Analytics Dashboard")

    if logs.empty:
        st.info("No processed queries yet. Analytics will appear after the first successful response.")
        return

    st.metric("Total Queries Processed", len(logs))

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.write("Count by Category")
        st.bar_chart(logs["category"].value_counts())

    with chart_columns[1]:
        st.write("Count by Priority")
        st.bar_chart(logs["priority"].value_counts())

    with st.expander("View recent logs"):
        st.dataframe(logs.tail(10), use_container_width=True)

def main():
    configure_page()
    apply_custom_styles()
    render_header()

    main_column, analytics_column = st.columns([2, 1], gap="large")
    with main_column:
        customer_query, submitted = render_query_form()
        render_response(customer_query, submitted)

    with analytics_column:
        render_analytics_dashboard()


if __name__ == "__main__":
    main()
