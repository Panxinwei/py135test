from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from IPython.display import display, Markdown
from langchain.memory import ConversationBufferMemory
import numpy as np
import warnings
import json
import os


def shap_to_json(features, shap_values, output_filepath, num_bins=10):
    """
    Convert the SHAP values into a detailed JSON structure, save it to a file, and print the filepath.

    Args:
        features (list): List of feature names.
        shap_values (numpy.ndarray): 2D array of SHAP values (samples x features).
        output_filepath (str): Filepath to save the output JSON file.
        num_bins (int): Number of bins for histogram calculations (default: 10).

    Returns:
        None
    """
    #Create an empty dictionary to hold the data for all features
    data = {}

    # Iterate over each feature and its corresponding SHAP values
    for i, feature in enumerate(features):
        #Extract SHAP values for the current feature across all samples
        feature_shap_values = shap_values[:, i]

        #Calculate key statistics for the SHAP values of the current feature
        mean_shap = float(np.mean(np.abs(feature_shap_values)))  #Mean of absolute SHAP values
        median_shap = float(np.median(feature_shap_values))     #Median SHAP value
        shap_25th = float(np.percentile(feature_shap_values, 25))  #25th percentile (Q1)
        shap_75th = float(np.percentile(feature_shap_values, 75))  #75th percentile (Q3)

        # Create a histogram of SHAP values for the current feature
        hist, bin_edges = np.histogram(feature_shap_values, bins=num_bins)
        #Organize histogram data into a list of dictionaries, each representing a bin
        bins = [
            {
                "bin_range": f"[{float(bin_edges[j]):.2f}, {float(bin_edges[j+1]):.2f}]",
                "count": int(hist[j])  #Number of SHAP values falling into this bin
            }
            for j in range(len(hist))
        ]

        #Add the calculated data for this feature to the dictionary
        data[feature] = {
            "mean_shap_value": mean_shap,
            "median_shap_value": median_shap,
            "shap_25th_percentile": shap_25th,
            "shap_75th_percentile": shap_75th,
            "binned_shap_values": bins
        }

    #Ranks the features by mean SHAP values from highest to lowest
    sorted_features = sorted(data.items(), key=lambda x: x[1]["mean_shap_value"], reverse=True)
    #Reconstruct the dictionary with features ranked by importance
    ranked_data = {feature: stats for rank, (feature, stats) in enumerate(sorted_features, start=1)}

    #Converts the ranked dictionary into a JSON string with proper indentation in the txt file
    json_data = json.dumps(ranked_data, indent=4)

    # Save the JSON string to the specified output file
    with open(output_filepath, "w") as file:
        file.write(json_data)

    #Prints the file path of the saved JSON file for the user's reference
    print(f"JSON data has been saved to: {os.path.abspath(output_filepath)}")
def load_shap_data(file_path):
    """
    Load SHAP data from a text file. Assumes the file contains valid JSON data.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    
    loader = TextLoader(file_path)
    try:
        documents = loader.load()
        shap_data = json.loads(documents[0].page_content)
        if not shap_data:
            raise ValueError("Error: The file contains no SHAP data.")
        return shap_data
    except json.JSONDecodeError:
        raise ValueError("Error: Failed to parse JSON data from the file.")


def initialize_chain():
    """
    Initialize the LLM chain with a memory component for conversation context.
    """
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "user_prompt"],
        template="Chat History: {chat_history}\nUser: {user_prompt}\nAssistant:"
    )
    return LLMChain(llm=llm, prompt=prompt_template, memory=memory)


def summarize_shap_data(shap_data):
    """
    Generate a Markdown summary of the SHAP data.
    """
    summary = "### **Summary of the SHAP Data**\n\n"
    summary += f"- **Number of features**: {len(shap_data)}\n\n"
    feature_names = list(shap_data.keys())
    summary += f"- **Feature names**: {', '.join(feature_names)}\n\n"

    feature_importance = [
        (feature, values.get("mean_shap_value", 0)) for feature, values in shap_data.items()
    ]
    feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

    summary += "### **Feature Importance (Ranked by Mean SHAP Values)**\n\n"
    summary += "| Rank | Feature Name         | Mean SHAP Value |\n"
    summary += "|------|----------------------|-----------------|\n"
    for rank, (feature, importance) in enumerate(feature_importance, 1):
        summary += f"| {rank} | {feature:<20} | {importance:.4f} |\n"
    return summary


def generate_prompts_with_llm(llm_chain, shap_data, recent_query=None):
    """
    Use the LLM to generate dynamic example prompts based on SHAP data and recent user input.
    Ensure at least one prompt continues from the user's recent query.
    """
    # Construct a detailed prompt for the LLM to generate example questions
    full_prompt = f"""
    You are an AI assistant helping users analyze SHAP (SHapley Additive exPlanations) values. Based on the SHAP data provided, generate 3 dynamic, insightful, and contextually relevant questions for the user.
    
    Here is the data for reference:
    {json.dumps(shap_data, indent=2)}

    Recent user context: "{recent_query or 'None'}"
    
    Ensure the questions meet the following criteria:
    1. At least one question should continue or expand upon the recent user query.
    2. Focus on actionable insights, feature importance, or optimization strategies.
    3. Ensure the questions are clear, relevant, and easy to understand.
    4. Return only the 3 questions as a list without additional commentary or introductory text.
    """
    # Run the constructed prompt through the LLM chain
    response = llm_chain.run({"user_prompt": full_prompt}).strip()

    # Extract the prompts as a list
    prompts = [line.strip("- ").strip() for line in response.split("\n") if line.strip()]

    # Ensure exactly 3 prompts
    if len(prompts) < 3:
        prompts += ["Additional prompt needed for consistency."] * (3 - len(prompts))
    prompts = prompts[:3]

    # If a continuation prompt is missing, explicitly add one
    if recent_query:
        continuation_prompt = f"Based on your previous query about '{recent_query}', what further analysis or action can be taken to refine insights?"
        if continuation_prompt not in prompts:
            prompts[-1] = continuation_prompt  # Replace the last prompt with the continuation prompt

    return prompts


def chat_with_gpt(prompt, llm_chain, shap_data):
    """
    Chat function that integrates the SHAP data into the user's prompt and provides a response.
    """
    full_prompt = f"""
  You are an AI assistant analyzing SHAP (SHapley Additive exPlanations) values. Based on the provided SHAP data and user's query, provide actionable insights and recommendations.
    1. Provide simple, actionable insights tailored to the user's question or context.
    2. Summarize findings with a concise, prioritized action plan.
    
    You should aim to:
    - Use plain, accessible language when explaining SHAP values and trends.
    - Avoid unnecessary technical detail; focus on the practical implications of the data.
    - Ensure all recommendations are immediately understandable and actionable.
    - Use quantitative measurements/reasoning to back your suggestions if possible.

    For every response:
    1. **Question:**
        - Reiterate the prompt and tell the user your understanding of the prompt.
        - Explain how you will answer the question.
    2. **Data-Driven Recommendations:**
       - Provide actionable insights tied directly to SHAP data.
       - Include quantitative reasoning based on SHAP data if possible.
       - Identify HOW to implement these recommendations and explain why these recommendations with the data.
    3. **Summary:** Prioritize the most impactful actions in a concise conclusion.

    Key Considerations:
    - Simplify SHAP value explanations and relate them to real-world outcomes.
    - Use SHAP distributions (bins and ranges) to support actionable advice.
    - Always tie recommendations back to the data provided, ensure that the data is used to back the recommendations.
    - Avoid technical jargon; focus on practical, measurable steps.

    Here is the data for reference:
    {json.dumps(shap_data, indent=2)}

    User's question: {prompt}
    """
    response = llm_chain.run({"user_prompt": full_prompt}).strip()
    return response


def start_chatbot(file_path):
    """
    Start the chatbot interface using SHAP data.
    """
    shap_data = load_shap_data(file_path)
    llm_chain = initialize_chain()

    # Display SHAP data summary
    summary = summarize_shap_data(shap_data)
    print("Chatbot: Welcome! Here is a summary of the SHAP data we will be working with:\n")
    display(Markdown(summary))

    # Generate and display initial prompts
    initial_prompts = generate_prompts_with_llm(llm_chain, shap_data)
    formatted_prompts = "\n\n\n### **Here are three example questions to get started:**\n\n"
    for i, prompt in enumerate(initial_prompts, 1):
        formatted_prompts += f"{i}. {prompt}\n\n"
    display(Markdown(formatted_prompts))
    
    print("\nChatbot: You can now ask your own questions. Type 'quit' to exit the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Goodbye!")
            break
        response = chat_with_gpt(user_input, llm_chain, shap_data)
        print("\nChatbot Response:")
        display(Markdown(response))

        # Suggest follow-up questions
        follow_up_prompts = generate_prompts_with_llm(llm_chain, shap_data, recent_query=user_input)
        formatted_follow_ups = "\n\n\n ### **Here are some additional questions you might want to ask:**\n\n"
        for i, prompt in enumerate(follow_up_prompts, 1):
            formatted_follow_ups += f"{i}. {prompt}\n\n"
        display(Markdown(formatted_follow_ups))