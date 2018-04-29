def read_file(filename):
    """READFILE reads a file and returns its entire contents
       file_contents = READFILE(filename) reads a file and returns its entire
       contents in file_contents
    """
    try:
        with open(filename, 'r') as file:
            return file.read()
    except IOError:
        print("can't open file", filename)
        return ''
