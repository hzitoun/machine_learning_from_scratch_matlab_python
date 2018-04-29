def get_vocab_list(file_name):
    """"GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
       cell array of the words
       vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
       and returns a cell array of the words in vocabList.
    """

    # Read the fixed vocabulary list
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            lines = [line.split('\t')[1].strip() for line in lines]
            return lines
    except IOError:
        print("can't open file", file_name)
        return []
