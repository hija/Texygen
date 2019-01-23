# coding=utf-8
import nltk

from nltk.tokenize import TweetTokenizer

T = TweetTokenizer()

def chinese_process(filein, fileout):
    with open(filein, 'r', encoding="utf-8") as infile:
        with open(fileout, 'w', encoding="utf-8") as outfile:
            for line in infile:
                output = list()
                line = T.tokenize(line)[0]#.word_tokenize(line)[0]
                for char in line:
                    output.append(char)
                    output.append(' ')
                output.append('\n')
                output = ''.join(output)
                outfile.write(output)


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        line = [item for sublist in [x.split() for x in line] for item in sublist]
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file, encoding='utf-8') as raw:
        for text in raw:
            text = T.tokenize(text.lower())#nltk.word_tokenize(text.lower())
            #text = [item for sublist in text for item in sublist]
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict

def get_relevant_word_set(word_set):
    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))
    return list(filter(lambda x: x not in stopWords, word_set))

def get_tokens_onehot_encoded(relevant_word_set, tokenized):
    ## Generates a one hot encoding
    one_hot_encoded_tokens = []
    for tokenized_str in tokenized:
        one_hot_encoded_token = [0] * len(relevant_word_set) # create one hot encoding
        for word in tokenized_str:
            if word in relevant_word_set: # If the current word is in the relevant words
                one_hot_encoded_token[relevant_word_set.index(word)] = 1
        one_hot_encoded_tokens.append(one_hot_encoded_token)
    return one_hot_encoded_tokens


def text_precess(train_text_loc, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc) # Gets the tokenized training file
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens) # A set of all possible words
    [word_index_dict, index_word_dict] = get_dict(word_set) # Mapping Word->Int and Int->Word

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len)) #sequence length == maximum length of all sequences
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    with open('save/eval_data.txt', 'w') as outfile: #create file eval_data
        # Create a file which maps test sequences -> int
        outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    ### RETURNS: Sequence Length of longest sequence in dataset + EOF index
    return sequence_len, len(word_index_dict) + 1, len(get_relevant_word_set(word_set))
