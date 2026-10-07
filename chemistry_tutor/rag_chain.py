"""
RAG chain — Chemistry Tutor persona.
Uses LangChain 1.x LCEL pipeline.
Supports Groq, IBM watsonx.ai, and OpenAI as LLM backends.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from chemistry_tutor.config import get_config, RETRIEVER_TOP_K

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Class 12 CBSE Chemistry teacher and mentor helping a student
prepare for the CBSE Board Exams (Chemistry code 043). Your teaching style is authentic CBSE Board standard, clear, rigorous, and exam-focused.

Official CBSE Class 12 Chemistry Syllabus:
- Unit 1: Solutions (Concentration, Colligative Properties, Van't Hoff Factor): ~7 Marks
- Unit 2: Electrochemistry (Galvanic Cells, Nernst Equation, Electrolysis): ~9 Marks
- Unit 3: Chemical Kinetics (Rate Laws, Activation Energy, Arrhenius Equation): ~7 Marks
- Unit 4: d- and f-Block Elements (Transition Metals, Lanthanoids): ~7 Marks
- Unit 5: Coordination Compounds (IUPAC, Isomerism, VBT, CFT): ~7 Marks
- Unit 6: Haloalkanes & Haloarenes (SN1/SN2, Elimination): ~6 Marks
- Unit 7: Alcohols, Phenols & Ethers: ~6 Marks
- Unit 8: Aldehydes, Ketones & Carboxylic Acids (Named Reactions): ~8 Marks
- Unit 9: Amines (Basicity, Diazonium Salts): ~6 Marks
- Unit 10: Biomolecules (Carbohydrates, Proteins, Nucleic Acids): ~7 Marks

Guidelines:
1. Always write balanced chemical equations with state symbols.
2. For named reactions (Aldol, Cannizzaro, Reimer-Tiemann, etc.), state: reactant → product → conditions → mechanism summary.
3. For electrochemistry, show full cell notation, half-reactions, and Nernst equation substitution step-by-step.
4. For coordination compounds, apply IUPAC naming rules precisely and draw/describe structural/optical isomers.
5. For numerical problems (molality, normality, EMF, rate constants), show every step with units.
6. Calibrate difficulty: 30% direct recall, 50% application, 20% HOTS/assertion-reason.
7. Cite and anchor explanations using the retrieved context from the student's study materials.

Context from study material:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_llm():
    """Instantiate the LLM based on dynamic configuration."""
    provider = get_config("LLM_PROVIDER", "groq").lower()

    if provider == "watsonx":
        from langchain_ibm import WatsonxLLM
        apikey = get_config("WATSONX_APIKEY") or get_config("WATSONX_API_KEY")
        return WatsonxLLM(
            model_id=get_config("WATSONX_MODEL_ID") or get_config("WATSONX_MODEL", "ibm/granite-3-8b-instruct"),
            url=get_config("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            apikey=apikey,
            project_id=get_config("WATSONX_PROJECT_ID"),
            params={"max_new_tokens": 8192, "temperature": 0.3, "repetition_penalty": 1.1},
        )
    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        api_key = get_config("GROQ_API_KEY")
        raw_model = get_config("GROQ_MODEL", "openai/gpt-oss-120b")
        model_aliases = {
            "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
            "llama-3.1-8b-instant": "openai/gpt-oss-20b",
            "llama3-70b-8192": "openai/gpt-oss-120b",
            "llama3-8b-8192": "openai/gpt-oss-20b",
            "mixtral-8x7b-32768": "openai/gpt-oss-120b",
        }
        model = model_aliases.get(raw_model, raw_model)
        if not api_key:
            raise ValueError("Missing `GROQ_API_KEY`. Configure it in .env or Streamlit Secrets.")
        return ChatOpenAI(
            model=model, api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3, max_tokens=8192, max_retries=3,
        )
    else:
        from langchain_openai import ChatOpenAI
        api_key = get_config("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing `OPENAI_API_KEY`. Configure it in .env or Streamlit Secrets.")
        return ChatOpenAI(
            model=get_config("OPENAI_MODEL", "gpt-4o"),
            api_key=api_key, temperature=0.3, max_tokens=8192,
        )


def build_rag_chain(vector_store):
    """Return a callable RAG chain with chat history support."""
    llm = _build_llm()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "fetch_k": 20},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(retriever.invoke(x["question"]))),
        )
        | RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )
    return chain


def convert_history(raw_history: list[dict]) -> list:
    """Convert Streamlit message dicts to LangChain message objects."""
    messages = []
    for msg in raw_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
