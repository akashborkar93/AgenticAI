from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class InputState(TypedDict):
    name: str
    greeting: str

#node-1
def ask_name(state: InputState) -> InputState:
    name = input("Bot: What is your name? ")
    return {'name': name, 'greeting': " "}

def greet_user(state: InputState) -> InputState:
    name = state.get('name') or 'there'
    greeting = f"Hello {name}, how may I help you !!"
    return {'name': name , 'greeting':greeting}

# Build Graph
def create_graph():
    graph= StateGraph(InputState)
    graph.add_node("branch_a", ask_name)
    graph.add_node("branch_b", greet_user)

    graph.add_edge(START, "branch_a")
    graph.add_edge("branch_a", "branch_b")
    graph.add_edge("branch_b", END)

    return graph.compile()

if __name__ == "__main__":
    app= create_graph()

    result = app.invoke({'name':" ",'greeting': " "})
    print(result)
    print("conversation complete")


 # --- Draw and save the graph image ---
    print("Generating graph image...")
    graph = app.get_graph()
    graph_path = "greeting_graph.png"
    graph.draw_mermaid_png(output_file_path=graph_path)
    print(f"Graph image saved at: {graph_path}")