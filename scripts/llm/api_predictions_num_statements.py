"""Prompt LLAMA to annotate the number of statements for each sentence in the test data"

from openai import OpenAI
import pandas as pd
import os

# API configuration
api_key = "YOUR_API_KEY"  # Replace with your API key
base_url = "https://chat-ai.academiccloud.de/v1"  # This runs over resources provided by the University of Göttingen. See: https://kisski.gwdg.de/
model = "meta-llama-3-70b-instruct"

# Start OpenAI client"meta-llama-3-70b-instruct"
client = OpenAI(api_key=api_key, base_url=base_url)


def get_response(sentence, annotation_guidelines):
    """Make an API request that specifies the model parameters and prompts."""
    chat_completion = client.chat.completions.create(
        seed=43,  # seed for reproducibility
        temperature=0.5,
        model=model,  # temperature for how much deterministic versus random the the model  is in its responses.
        messages=[
            {  # System prompts are used to set the "persona" of a model.
                "role": "system",
                "content": "You are an expert in German Easy Language.",
            },
            {
                "role": "user",
                "content": "How many statements does the sentence below have? For your decisions rely on the annotation guidelines provided below. Provide your output in form of an integer. Return nothing but that integer. Think step by step. ",
            },
            {
                "role": "user",
                "content": "Three example sentences: sentence: Eine sehr bekannte Alchemisten war Maria die Jüdin statements:1; sentence: Im Jahr 1986 hat die UNESCO gesagt: Die Alt-stadt von Aleppo ist jetzt Welt-kultur-erbe. statements:4 ; sentence:Die Mutter von Bell heißt Eliza Symonds Bell. statements: 1 ;   The sentence you should annotate is the following:",
            },
            {"role": "user", "content": sentence},
            {"role": "user", "content": "Annotation guidelines:"},
            {"role": "user", "content": annotation_guidelines},
        ],
    )

    output = chat_completion.choices[0].message.content
    return output


def main():

    # Define components of file paths
    data_dir = "data"
    test_filename = "test.csv"
    llm_dir = "LLM"
    annotation_guidelines_filename = "annotation_guidelines_cleaned.txt"

    # Create the full file paths, independent of system.
    filepath_testdata = os.path.join(data_dir, test_filename)
    filepath_annotation_guidelines = os.path.join(llm_dir, annotation_guidelines_filename)

    with open(
        filepath_annotation_guidelines, "r"
    ) as file:  # note that you have to remove all links and newlines in the annotation guidelines or there will be issues with preprocessing.
        annotation_guidelines = file.read()

    # Make API requests for every sentence in the test dataset.
    df_test_data["LLAMA predictions"] = df_test_data["phrase"].apply(
        get_response, args=(annotation_guidelines,)
    )

    file_out = "LLAMA_results_number_of_statements.csv"

    df_test_data.to_csv(file_out, index=False)
