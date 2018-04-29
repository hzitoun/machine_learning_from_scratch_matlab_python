from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from getVocabList import get_vocab_list

import re

# uncomment if you want to check fo newer version
# import nltk
# nltk.download('punkt')


def regexprep(contents, regex, replace_value):
    return re.sub(regex, replace_value, contents)


def process_email(email_contents, vocab_file_name):
    """Pre-processes a the body of an email and
       returns a list of word_indices
       word_indices = PROCESSEMAIL(email_contents) preprocesses
       the body of an email and returns a list of indices of the
       words contained in the email.
    """

    # Load Vocabulary
    vocab_list = get_vocab_list(vocab_file_name)

    # Init return value
    word_indices = []

    # ========================== Pre-process Email ===========================

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = regexprep(email_contents, r'<[^<>]+>', ' ')

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = regexprep(email_contents, r'[0-9]+', 'number')

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = regexprep(email_contents, r'(http|https)://[^\s]*', 'httpaddr')

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = regexprep(email_contents, r'[^\s]+@[^\s]+', 'emailaddr')

    # Handle $ sign
    email_contents = regexprep(email_contents, r'[$]+', 'dollar')

    # get rid of any punctuation
    email_contents = regexprep(email_contents, r'[^\w\s]', '')

    # remove \n
    email_contents = regexprep(email_contents, r'\n', '')

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n', email_contents)

    stemmer = PorterStemmer()

    # Tokenize
    for token in word_tokenize(email_contents):
        # Remove any non alphanumeric characters
        word = regexprep(token.strip(), '[^a-zA-Z0-9]', '')

        # Stem the word
        word = stemmer.stem(word)

        # Skip the word if it is too short
        if len(word) < 1:
            continue

        # append index
        try:
            word_indices.append(vocab_list.index(word) + 1)  # add one because training set indexes start from 1
        except ValueError:
            continue

    return word_indices

