import random
import re
from decimal import Decimal
from collections import Counter
import pandas as pd

def preprocess_text(text):
    """Preprocesses the text.
    
    Args:
        text (str): Text to be processed.
    
    Returns:
        str: Processed text.
    """
    # Add starting and ending tags to all the sentences
    tagged = re.sub('\.', " </s> <s>", text)
    tagged = "<s> " + tagged[:-4]

    # Convert all characters to lowercase
    tagged_lower = tagged.lower()

    # Define the pattern to match punctuation marks
    punctuation_pattern = r'[,.!;?]'
    
    # Remove punctuation marks using regex
    tagged_cleaned = re.sub(punctuation_pattern, '', tagged_lower)

    return tagged_cleaned

def create_ngrams(text, num):
    """Creates n-grams from the given text.
    
    Args:
        text (str): Text to generate n-grams from.
        num (int): Number of words per n-gram.
    
    Returns:
        list: List of n-grams.
    """
    # Split the text on space and trim extra whitespaces
    splitted = [x.strip() for x in text.split(" ")]

    # Create n-grams based on the input using a list comprehension
    ngrams = [' '.join(splitted[i:i+num]) for i in range(len(splitted) - num + 1)]

    return ngrams

def generate_random_sentence(tokens, size):
    """Generates a random sentence.
    
    Args:
        tokens (set): Set of words to be used as tokens.
        size (int): Size of the desired output.
    
    Returns:
        str: Randomly generated sentence.
    """
    # Randomly choose 'size' different samples from the set of unique words
    sentence_list = random.choices(list(tokens), k=size)

    # Create the sentence from the list and add starting and ending tags
    return "<s> " + " ".join(sentence_list) + " </s>"

corpus = "Language processing, also known as natural language processing (NLP), \
is a field of artificial intelligence that focuses on the interaction between \
computers and human language. It involves the development of algorithms and \
models to enable computers to understand, interpret, and generate human language. \
NLP encompasses a wide range of tasks, including speech recognition, sentiment \
analysis, machine translation, information extraction, and text generation. \
By leveraging techniques from linguistics, computer science, and statistics, \
language processing aims to bridge the gap between human communication and \
machine understanding."

output_text = ''

# Count the number of words in the text
word_count = len(corpus.split())
word_nums = f"\nThe text contains {word_count} words.\n"
print(word_nums)
output_text += word_nums

# Process the input text to prepare it for generating n-grams
processed_text = preprocess_text(corpus)  

# Generate unigrams from the processed text
unigrams = create_ngrams(processed_text, 1)  

# Generate bigrams from the processed text
bigrams = create_ngrams(processed_text, 2) 

# Count the total number of unigrams and bigrams
total_unigrams = len(unigrams)
total_bigrams = len(bigrams)

# Print the counts
print("Total number of unigrams:", total_unigrams)
print("Total number of bigrams:", total_bigrams, '\n')

output_text += "\nTotal number of unigrams: " + str(total_unigrams) + "\n"
output_text += "Total number of bigrams: " + str(total_bigrams) + "\n"

# Count the number of unique unigrams and bigrams
unique_unigrams = len(set(unigrams))
unique_bigrams = len(set(bigrams))

output_text += "\nNumber of unique unigrams: " + str(unique_unigrams) + "\n"
output_text += "Number of unique bigrams: " + str(unique_bigrams) + "\n"

# Add all the unigrams to a set to have a list of unique words
tokens = {token for token in unigrams if token not in {"<s>", "</s>"}}

# Count the length of the set to get the total number of unique tokens
token_count = len(tokens)

# Generate a random sentence using the tokens and a random length between 2 and 5
sentence = generate_random_sentence(tokens, random.randint(2, 5))

# Create unigrams for the generated sentence
sentence_unigrams = create_ngrams(sentence, 1)

# Create bigrams for the generated sentence
sentence_bigrams = create_ngrams(sentence, 2)

# Set this to print all rows of tables
pd.set_option('display.max_rows', None)

# Count the number of unigrams and bigrams using Counter
unigram_counts = Counter(unigrams)
bigram_counts = Counter(bigrams)

# Find the maximum length of strings in the Unigram and Bigram columns
max_unigram_length = max(len(unigram) for unigram in unigram_counts.keys())
max_bigram_length = max(len(bigram) for bigram in bigram_counts.keys())

# Create a table to display the counts
unigram_table = pd.DataFrame({'Unigram': list(unigram_counts.keys()), 'Count': list(unigram_counts.values())})
bigram_table = pd.DataFrame({'Bigram': list(bigram_counts.keys()), 'Count': list(bigram_counts.values())})

# Get the maximum length of each column
unigram_max_length = max(unigram_table['Unigram'].astype(str).map(len).max(), len('Unigram'))
count_max_length = max(unigram_table['Count'].astype(str).map(len).max(), len('Count'))
bigram_max_length = max(bigram_table['Bigram'].astype(str).map(len).max(), len('Bigram'))

# Calculate the total width of the tables
unigram_total_width = unigram_max_length + count_max_length + 16
bigram_total_width = bigram_max_length + count_max_length + 16

# Print the Unigram Counts table with separators
print('-' * unigram_total_width)
header = f"| {'':^6} | {'Unigram':^{unigram_max_length}} | {'Count':^{count_max_length}} |"
print(header)
print('-' * unigram_total_width)

output_text += '\n' + '-' * unigram_total_width + '\n' + header + '\n' + '-' * unigram_total_width + '\n'

for i, row in unigram_table.iterrows():
    index = str(i).center(6)
    unigram = str(row['Unigram']).center(unigram_max_length)
    count = str(row['Count']).center(count_max_length)
    print(f"| {index} | {unigram} | {count} |")
    print('-' * unigram_total_width)
    output_text += f"| {index} | {unigram} | {count} |\n"
    output_text += '-' * unigram_total_width + "\n"

# Print the Bigram Counts table with separators
print("\n" + '-' * bigram_total_width)
header = f"| {'':^6} | {'Bigram':^{bigram_max_length}} | {'Count':^{count_max_length}} |"
print(header)
print('-' * bigram_total_width)

output_text += '\n' + '-' * bigram_total_width + '\n' + header + '\n' + '-' * bigram_total_width + '\n'

for i, row in bigram_table.iterrows():
    index = str(i).center(6)
    bigram = str(row['Bigram']).center(bigram_max_length)
    count = str(row['Count']).center(count_max_length)
    print(f"| {index} | {bigram} | {count} |")
    print('-' * bigram_total_width)
    output_text += f"| {index} | {bigram} | {count} |\n"
    output_text += '-' * bigram_total_width + "\n"

# Convert the set of unique words to a list
unique_words = list(set(processed_text.split()))

# Initialize an empty matrix with rows and columns for each unique word
matrix = pd.DataFrame(index=unique_words, columns=unique_words)

# Fill the matrix with zeros
matrix = matrix.fillna(0)

# Update the matrix with the counts of each bigram occurrence
for bigram in bigrams:
    word1, word2 = bigram.split()
    matrix.loc[word1, word2] += 1

# Print the Bigram Matrix
print("\nBigram Matrix:\n")
print(matrix)

# Calculate the sentence probability of generating the generated sentence
total_probability = 1
# Looping through the bigrams of the sentence
for i in range(len(sentence_bigrams)):
    # Counting occurrences of the first word in each bigram in the corpus's unigrams
    unigram_count  = unigrams.count(sentence_unigrams[i])
    # Counting occurrences of the bigram in the corpus's bigrams
    bigram_count = bigrams.count(sentence_bigrams[i])
    # Calculating the probability with add-1 smoothing
    probability = (bigram_count + 1) / (unigram_count + token_count)
    # Updating the total probability by multiplying it with each calculated bigram probability
    total_probability *= probability

# Print the generated sentence and its probability
print("\nGenerated Sentence:", sentence)
print("Sentence Probability:", f"{Decimal(total_probability * 100):.3E}%\n")

# Write the output text to a file
with open("output.txt", "w") as file:
    file.write(output_text)
    file.write("\nGenerated Sentence: " + sentence)
    file.write("\nSentence Probability: " + f"{Decimal(total_probability * 100):.3E}%")

# Add the matrix to the output text
output_text += "\n\nBigram Matrix:\n\n" + str(matrix)

print('\nOutput text:\n')
print(output_text)
