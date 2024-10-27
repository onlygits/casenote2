import os
from openai import OpenAI
import streamlit as st
from datetime import date
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date  
import hmac
__import__('pysqlite3')
import sys
import sqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



# Load environment variables
openai_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_key)

# Model objectives
model_objectives = {
    "Parent induction session": """The objective of the session is for the caseworker to conduct a home visit and introduce themselves to the family or caregivers. It is also to provide information on the processes of the youth home.""",
    "Introduction session with youth": """The objective of the session is to introduce the youth to the caseworker and provide an overview of the programs they will be involved in."""
}

# Model notes for what was discussed
model_discussed = {
    "Parent induction session": """Caseworker visited the family's home and discussed how the youth has been doing, their journey in the home (e.g., the tiered reward system), the various programs provided, rules and regulations to follow, and obtained signed parental consent on all necessary parent induction forms.""",
    "Introduction session with youth": """The youth is briefed by the caseworker on the rules and regulations of the youth home, and the session is an opportunity for the caseworker to understand any initial concerns or questions."""
}

# Follow-up session actions
follow_up_sessions = {
    "Parent induction session": """Caseworker to conduct a family session with both family and the youth to facilitate bonding and relationship building. Caseworker will continuously update the family on the youth's progress and address any needs they might present.""",
    "Introduction session with youth": """One-on-one caseworker session to explore the youth's background and reasons for admission, which will be used to formulate the youth's Individual Case Plan (ICP)."""
}


# Step 1: Use AI to Combine and Paraphrase the Discussion Section
def generate_combined_discussion(model_note, discussion_input):
    prompt = f"""
    As a caseworker, combine the model discussion with the additional discussion input delimited by "###". You MUST include all information provided and do not repeat similar information:

    ###
    Model Discussion:
    {model_note}\n
    
    Additional Input:
    {discussion_input}\n
    ###

    Ensure the output flows smoothly and uses complete, professional sentences.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# Step 2: Use AI to Combine and Paraphrase the Follow-Up Section
def generate_combined_followup(model_followup, followup_input):
    prompt = f"""
    As a caseworker, combine the model follow-up actions with the additional follow-up actions delimited by "###". You MUST include all information provided do not repeat similar information:

    ###
    Model Follow-Up Actions:
    {model_followup}\n

    Additional Follow-Up Actions:
    {followup_input}\n
    ###

    Provide a single, cohesive follow-up plan, with professional sentences.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# Step 3: Generate the Final Case Note
def generate_case_note(session_type, participants, session_date, discussion_input, followup_input):
    model_note = model_discussed[session_type]
    model_followup = follow_up_sessions[session_type]
    model_objective = model_objectives[session_type]

    # Generate the combined discussion and follow-up using AI
    final_discussion = generate_combined_discussion(model_note, discussion_input)
    final_followup = generate_combined_followup(model_followup, followup_input)

    # Create the final case note
    prompt = f"""
    You are a caseworker documenting case notes for a "{session_type}". Below are the session details delimited by "###". You MUST include all information provided in the following format:

    ###
    **Session Title:** {session_type}\n
    **Date:** {session_date}\n
    **Who were Present:** {participants}\n
    **Objective of the Session:** 
    {model_objective}\n
    **What was Discussed:** 
    {final_discussion}\n
    **Follow-up Actions:** 
    {final_followup}\n
    ###

    As a caseworker, ensure your final output is written in third person, in a formal tone. Information should be consistent across segments. For example, the people present, objective of the session, what was discussed and follow-up should all align. Finally be less florid.
    
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content


### Code for the Resource Finder ###
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import uuid
from langchain_chroma import Chroma
import pandas as pd
import streamlit as st


# Load environment variables
llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_key)

# Process pdf document
loader = PyPDFLoader("strategies_and_interventions.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                               chunk_overlap=200,
                                               length_function=len,
                                               separators=["\n\n","\n"," "])

chunks = text_splitter.split_documents(pages)

# create embeddings
def get_embedding_function():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=openai_key
    )
    return embeddings                                      
embedding_function = get_embedding_function()



def create_vectorstore(chunks, embedding_function, vectorstore_path):
    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    # Chroma database to store vector embeddings
    vectorstore = Chroma.from_documents(documents=unique_chunks,
                                        ids=list(unique_ids),
                                        embedding=embedding_function,
                                        persist_directory=vectorstore_path)

    return vectorstore


# Create the vectorstore
vectorstore = create_vectorstore(chunks=chunks, embedding_function=embedding_function, vectorstore_path="vectorstore_chroma")

# Load the vectorstore (assuming it's needed immediately after creation might be redundant unless it's for a different session or purpose)
vectorstore = Chroma(persist_directory="vectorstore_chroma", embedding_function=embedding_function)

# Create a retriever and get relevant chunks
retriever = vectorstore.as_retriever(search_type="similarity")


# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for caseworkers and are an expert in question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don't know the answer,
say "I don't know". DO NOT MAKE ANYTHING UP.

{context}

###

Answer the question based on the above context: {question}
"""


#create prompt
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

#Using Langchain Expression Language

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Desired output structure
class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning"""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")

class ExtractedInfoWithSources(BaseModel):
    """Extracted information from the resource"""
    summary: AnswerWithSources


def set_css():
    st.markdown("""
        <style>
        .dataframe {
            font-size: 18px; /* Larger font size for better readability */
            border-collapse: collapse;
            margin-bottom: 20px; /* Adds space below the dataframe */
        }
        .dataframe th {
            background-color: #f0f2f6;
            color: #333;
        }
        .dataframe td, .dataframe th {
            min-width: 150px; /* Wider cells for more content */
            max-width: 600px; /* Limiting max width */
            text-align: left;
            padding: 15px; /* More padding for readability */
            border: 1px solid #DDD;
            white-space: normal !important; /* Forces text wrapping */
            word-wrap: break-word;
            vertical-align: top; /* Aligns text to the top of the cell */
        }
        </style>
        """, unsafe_allow_html=True)
    

# Here you should configure your retriever, formatting functions, and the LangChain model setup.
def setup_rag_chain():
    # Assume retriever, format_docs, prompt_template, and llm are defined elsewhere and correctly configured
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfoWithSources))

# Initialize the RAG chain
rag_chain = setup_rag_chain()


# Function to check if the answer is meaningful
def is_meaningful_answer(answer):
    # Define phrases that indicate an irrelevant or unknown answer
    uninformative_phrases = ["i don't know", "unknown", "cannot determine", "no information"]
    return not any(phrase in answer.lower() for phrase in uninformative_phrases)




### Password ###
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


### Streamlit App ###


def set_page(page_name):
    st.session_state.current_page = page_name

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Case Note Generator"

st.sidebar.title("Navigation")

pages = ["Case Note Generator", "Resource Search Engine", "About Me", "Methodology"]
for page in pages:
    if st.sidebar.button(page):
        set_page(page)

selected_page = st.session_state.current_page

if selected_page == "Case Note Generator":
    st.title("Case Note Generator")

    # Disclaimer Section
    with st.expander("IMPORTANT NOTICE", expanded=False):
        st.write("""
        Please do not enter any personally identifying information about the youths or their families in the form below. This includes names, identification numbers, or specific personal details.
        
        This web application is developed as a proof-of-concept prototype. 
        The information provided here is **NOT intended for actual usage** and should not be relied upon 
        for making any decisions, especially those related to financial, legal, or healthcare matters.
        
        Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. 
        You assume full responsibility for how you use any generated output.
        
        Always consult with **qualified professionals** for accurate and personalized advice.
        """)

    with st.form("Case Note Form"):
        session_type = st.selectbox("Select session type", list(model_discussed.keys()))
        participants = st.text_input("Who were present?")
        session_date = st.date_input("Date of session", value=date.today())
        discussion_input = st.text_area("Discussion details")
        followup_input = st.text_area("Follow-up actions")
        submit_button = st.form_submit_button("Generate Case Note")

    if submit_button:
        st.session_state.case_note = generate_case_note(
            session_type, participants, session_date, discussion_input, followup_input
        )
        st.subheader("Generated Case Note")
        st.markdown(st.session_state.case_note)

elif selected_page == "Resource Search Engine":

    st.title("Search Resource Database for Working with Youths")
    set_css()

    suggested_queries = [
        "Select a suggested query",  # Default option
        "What are the practice strategies for working with youth's families?",
        "How to handle confidentiality?",
        "What are some effective interventions or programs for youths?"
    ]

    # Using a form to handle inputs and submission
    with st.form("Resource_Query_Form"):
        # Dropdown for selecting a suggested query
        selected_query = st.selectbox("Try one of these suggested queries:", suggested_queries, index=0)
        # Displaying 'OR' with smaller and less obtrusive styling
        st.markdown("""
            <div style='text-align: center; color: gray; margin-top: 10px; margin-bottom: 10px;'>
                OR
            </div>
            """, unsafe_allow_html=True)
        # Text input for manual entry or selected query
        query = st.text_input("Enter your query", value="" if selected_query == "Select a suggested query" else selected_query)

        # Submit button for the form
        submitted = st.form_submit_button("Search")

    if submitted and query not in ["", "Select a suggested query"]:
        try:
            structured_response = rag_chain.invoke(query)
            if not isinstance(structured_response, ExtractedInfoWithSources):
                answer_with_sources = AnswerWithSources(
                    answer=structured_response['summary']['answer'],
                    sources=structured_response['summary']['sources'],
                    reasoning=structured_response['summary']['reasoning']
                )
                response_model = ExtractedInfoWithSources(summary=answer_with_sources)
            else:
                response_model = structured_response

            if is_meaningful_answer(response_model.summary.answer):
                response_data = {
                    "answer": response_model.summary.answer,
                    "source": response_model.summary.sources,
                    "reasoning": response_model.summary.reasoning
                }
                df = pd.DataFrame([response_data])
                st.table(df)

                # Suggested Next Steps
                st.subheader("Suggested Next Steps:")
                suggestions = llm.invoke(
                    f"""Based on the user query: '{query}' and the answer provided below from our resources for caseworkers working with youths in our residential home, suggest appropriate next steps for the caseworker.
                    The answer is delimited by '###'.
                    ###
                    {response_model.summary.answer}
                    ###
                    """
                )
                st.write(suggestions.content)
            else:
                st.warning("The query is unrecognized or irrelevant. Please try rephrasing")
        except Exception as e:
            st.error(f"An error occurred: {e}")


elif selected_page == "About Me":

    st.title("About Me")
    
    # Introduction or summary
    st.markdown("""
    ### Project Summary
    This project includes a case note generator as well as a resource search engine. The first use case aims to address the challenges faced by caseworkers due to time constraints and high volume of administrative tasks. The second use case aims to enhance interventions provided by caseworkers.
    """)

    # Problem Statement
    st.markdown("""
    ### Problem Statement
    Due to time constraints and ongoing operational needs, it is difficult for caseworkers at Youth homes to continuously document their sessions in the online system. This might lead to a backlog of case notes to be input into the system while they might have their notes written elsewhere or in point form. Additionally, based on what comes up during the session, caseworker might want to quickly look up relevant resources since currently there is not one designated place and there are a lot of documents to sieve through.
    """)

    # Proposed Solution
    st.markdown("""
    ### Proposed Solution
    Since there are many repetitive sessions conducted in the Youth homes, such as "parent induction sessions", "modules in the General Skills Programme" or "introduction session with youth", the general content of these sessions will be largely similar. With minimal inputs from the caseworkers on how the session went, the general body of the case notes can be generated formally by LLM by referencing examples and user inputs. Users will be warned not to input any identifying data to avoid data privacy issues.
    """)

    # Impact
    st.markdown("""
    ### Impact
    These solutions will save time needed by caseworkers to have their case notes written in a formal manner that could be uploaded into the system. There is a minimum number of case notes required for each youth per month, and each caseworker handles multiple cases. Considering the operational needs of the home, and the various other administrative tasks needed to be completed by case workers, this case note generator might be useful in cutting down administrative time. The saved time could then be spent engaging with the youth or their families instead. Similar benefits can be gained from the resource search engine. \n About **20** caseworkers stand to benefit from these use cases. \n These caseworkers are working with approximately **140** youths and their families currently. The numbers are expected to rise after the CYPA amendments. 
    """)

    # Project Sponsors & Users
    st.markdown("""
    ### Project Sponsors & Users
    The project would be used by about 20 case workers covering all the youths in both MSF YRS Homes, Singapore Girls' Home and Singapore Boys' Home.
    """)

    # Data Sources
    st.markdown("""
    ### Data Sources
    1. **Case Notes Generator:** Dummy model case notes based on session types + user inputs\n
    2. **Resource Search Engine:** 1 dummy resource found online in the form of a pdf document\n
    """)

    # Data Classification & Sensitivity
    st.markdown("""
    ### Data Classification & Sensitivity
    **Classification:** Restricted / Non-sensitive\n
    However, currently, only dummy data is used.\n
    """)

    # Project Source Code
    st.markdown("""
    ### Project Source Code
    [GitHub Repository](https://github.com/onlygits/casenote2)
    """)

elif selected_page == "Methodology":

    st.title("Methodology")
    
    # General description of the methodology
    st.markdown("""
    This section describes the methodologies used in the development of the two main components of this application:\n\n
    
    1. **Case Notes Generator**: Utilizes GPT 3.5 model to generate detailed case notes based on minimal user inputs.\n
    2. **Resource Search Engine**: Leverages on langchain and GPT 3.5 to retrieve and display resources relevant to caseworkers.\n
    """)
    
    # Displaying the methodology diagram for Case Notes Generator
    st.header("Case Notes Generator")
    st.image("casenote_method.png", caption="Process flow for the Case Notes Generator")
    
    # Displaying the methodology diagram for Resource Search Engine
    st.header("Resource Search Engine")
    st.image("resourcefinder_method.png", caption="Process flow for the Resource Search Engine")
