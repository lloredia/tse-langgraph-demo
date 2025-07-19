from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

def run_qa_node(state):
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question as clearly as possible: {question}"
    )
    llm = ChatOpenAI(temperature=0)
    chain = prompt | llm
    result = chain.invoke({"question": question})
    return {"question": question, "answer": result.content}

graph = StateGraph()
graph.add_node("qa_node", run_qa_node)
graph.set_entry_point("qa_node")
graph.add_edge("qa_node", END)

qa_app = graph.compile()
