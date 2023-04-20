import json
import token as tk
from io import StringIO
from tokenize import generate_tokens


def read_labels(dataset_file, tag='Linear_Formula'):
    labels = []
    for i, pair in enumerate(json.load(open(dataset_file, 'r'))):
        label = str(pair[tag])
        labels.append(label.lower())
    return labels


def get_encoded_code_tokens(code):
    code = code.strip()
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    indent_level = 0
    new_line = False

    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        #print(toknum, tokval, srow, scol, erow, ecol)
        if toknum == tk.NEWLINE:
            tokens.append(' ')
            new_line = True
        elif toknum == tk.INDENT:
            indent_level += 1
            # new_line = False
            # for i in range(indent_level):
            #     tokens.append('#INDENT#')
        elif toknum == tk.STRING:
            tokens.append(tokval.replace('\t', ' ').replace('\r\n', ' ').replace('\n', ' '))
        elif toknum == tk.DEDENT:
            indent_level -= 1
            # for i in range(indent_level):
            #     tokens.append('#INDENT#')
            # new_line = False
        else:
            tokval = tokval.replace('\n', ' ')
            if new_line:
                for i in range(indent_level):
                    tokens.append(' ')

            new_line = False
            tokens.append(tokval)

    # remove ending None
    if len(tokens[-1]) == 0:
        tokens = tokens[:-1]

    if '\n' in tokval:
        pass

    return tokens
