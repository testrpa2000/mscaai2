#!pip install spacy
#!python -m spacy download en_core_web_sm

import spacy

class LanguageParser:
    def __init__(self):
        # Load the spaCy English language model
        self.nlp = spacy.load("en_core_web_sm")

    def parse_query(self, query):
        # Process the input query using spaCy
        doc = self.nlp(query)

        # Example: Extracting verbs from the parsed query
        verbs = [token.text for token in doc if token.pos_ == "VERB"]

        return verbs

# Example Usage
if __name__ == "__main__":
    # Create an instance of the LanguageParser
    parser = LanguageParser()

    # User enters a query
    user_query = input("Enter your query: ")

    # Parse the query
    parsed_result = parser.parse_query(user_query)

    # Display the parsed result
    print("\nParsed Result:")
    if parsed_result:
        for i, verb in enumerate(parsed_result, 1):
            print(f"{i}. {verb}")
    else:
        print("No verbs found in the query.")

# '''
# Multiline
# Output

# Enter your query: 

# Artificial intelligence, or AI, refers to the simulation of human intelligence by
# software-coded heuristics. Nowadays this code is prevalent in everything from cloud-based, 
# enterprise applications to consumer apps and even embedded firmware.

# Parsed Result:
# 1. refers
# 2. coded
# 3. based
# 4. embedded

# Importing spaCy: The code begins with importing the spaCy library, which is a powerful natural language processing (NLP) library in Python.
# Defining the LanguageParser class: The LanguageParser class is defined, which will encapsulate the functionality to parse natural language queries.
# __init__ method: This method initializes an instance of the LanguageParser class. It loads the English language model provided by spaCy (en_core_web_sm) using spacy.load() and assigns it to the self.nlp attribute. This model will be used for processing natural language text.
# Defining the parse_query method: The parse_query method is defined to parse the user's input query.
# parse_query method: This method takes a single argument query, representing the user's input text. Inside this method:
# self.nlp(query): It processes the input query using the spaCy NLP pipeline, which tokenizes the text, performs part-of-speech tagging, dependency parsing, etc., and returns a Doc object.
# [token.text for token in doc if token.pos_ == "VERB"]: It iterates through each token in the parsed document (doc). For each token, it checks if the part-of-speech (POS) tag is "VERB". If it is, it adds the token's text to the verbs list.
# Finally, it returns the list of verbs extracted from the input query.
# Example usage: The code includes an example usage section that demonstrates how to use the LanguageParser class.
# if __name__ == "__main__":: This conditional block ensures that the code inside it only runs if the script is executed directly, not when imported as a module.
# It creates an instance of the LanguageParser class.
# It prompts the user to input a query.
# It calls the parse_query method of the LanguageParser instance to parse the user's query.
# It displays the parsed result, printing each extracted verb along with its index. If no verbs are found, it prints a message indicating that.