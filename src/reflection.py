from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from typing import Annotated, TypedDict  
from langgraph.graph.message import add_messages 
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are a helpful ai assistant that generates blog content based on user 
     input. You will receive feedback from reflection and based on the feedback 
     you will improve the blog post. The blog post should:
        - Have a strong hook
        - Provide useful information
        - Be technically accurate
        - Be concise
        - Avoid generic statements
        - Encourage engagement
        
     """
     ),
    MessagesPlaceholder(variable_name="messages")
    ]
)


reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are a helpful ai assistant that generates critics based on the 
     generated blog post. You will provide detail feedback on the blog post and
     suggest improvements based on the following areas:
        
        - Technical accuracy
        - Value
        - Clarity
        - Specificity
        - Engagement
        - Originality

        Provide specific recommendations for improvement.
    """
    ),
    MessagesPlaceholder(variable_name="messages")
    ]
)


quality_prompt = ChatPromptTemplate.from_messages([
    
    ("system", 
    
    """
    You are a helpful ai assistant that evaluates the quality of the generated 
    blog post about data science, machine learning and artificial intelligence.
    Evaluate the proposed post using these criteria:

        1. Value:
           Does the reader learn something useful?

        2. Technical accuracy:
           Are the contents technically correct?

        3. Clarity:
           Is the post easy to understand?

        4. Specificity:
           Does it provide concrete insight rather than generic AI advice?

        5. Engagement:
           Would the target audience have a reason to like,
           bookmark, or reply?

        6. Originality:
           Does it present an interesting perspective?

        Give each category a score from 1 to 10.

        Approve the post only if the overall quality is at least 8.

        Return structured output.
        
        """
    ),
    MessagesPlaceholder(variable_name="messages")
   ]
)


class BlogPostQuality(BaseModel):

    value: int = Field(ge=1, le=10)
    technical_accuracy: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    specificity: int = Field(ge=1, le=10)
    engagement: int = Field(ge=1, le=10)
    originality: int = Field(ge=1, le=10)
    overall: float = Field(ge=1, le=10)
    approved: bool
    reason: str

llm = ChatOpenAI(model_name="gpt-4o")
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

quality_llm = llm.with_structured_output(BlogPostQuality)
quality_chain = quality_prompt | quality_llm


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    quality: BlogPostQuality
    iteration: int

graph = StateGraph(AgentState)

def generate_blog_post(state):
    response = generation_chain.invoke({"messages": state["messages"]})
    
    return {"messages": [response],
            "iteration": state["iteration"] + 1
            }
    
def reflect_on_blog_post(state):
    response = reflection_chain.invoke({"messages": state["messages"]})
    
    return {"messages": [response]}

def quality_check(state):
    response = quality_chain.invoke({"messages": state["messages"]})
    
    return {"quality": response}

def should_continue(state):
    if state["quality"].approved:
        return END 
    if state["iteration"] >= 3:
        return END 
    
    return "reflect_on_blog_post" 


graph.add_node("generate_blog_post", generate_blog_post)
graph.add_node("reflect_on_blog_post", reflect_on_blog_post)
graph.add_node("quality_check", quality_check)
graph.set_entry_point("generate_blog_post")
graph.add_edge("generate_blog_post", "quality_check")
graph.add_edge("reflect_on_blog_post", "generate_blog_post")
graph.add_conditional_edges("quality_check", should_continue,
                            {
                                END: END,
                                "reflect_on_blog_post": "reflect_on_blog_post"
                            }
                        )

graph = graph.compile()

response=graph.invoke({
    "messages": [
        HumanMessage(content="Write a blog post on the importance of reflection agent within 300 words.")
        ],
    "iteration": 0
    })

print(f"Final Blog Post: {response['messages'][-1].content}")

