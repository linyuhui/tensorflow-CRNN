"""Usage:
Build vocab:
python vocab.py \
--operation build --output_dir datasets --name vocab

Read vocab: char - code
python vocab.py \
--operation read --input_dir datasets --name vocab --char_to_code
"""

import argparse
import codecs
import os
import re
import logging
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--name', type=str, default='', help='Name of vocab file (not include suffix).')
parser.add_argument('--operation', required=True, choices=['build', 'read'])
parser.add_argument('--char_to_code', action='store_true')
flags = parser.parse_args()

charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z']


def build_vocab():
    with codecs.getwriter('utf-8')(open(os.path.join(flags.output_dir, flags.name + '.txt'), 'wb')) as fw:
        fw.write('0\n')
        fw.write(str(len(charset) + 1) + '\t<nul>\n')  # EOC
        for i in range(1, len(charset) + 1):
            fw.write(str(i) + '\t' + charset[i - 1] + '\n')
    print(' Done.')


def read_char_vocab(filename, null_character=u'\u2591'):
    """
    Return: {char: code}
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    char_vocab = {}
    with codecs.getreader('utf-8')(open(filename, 'rb')) as fr:
        for i, line in enumerate(fr):
            m = pattern.match(line)
            if m is None:
                if i == 0:
                    char_vocab[' '] = 0
                    continue
                logging.warning('incorrect vocab file. line #%d: %s', i, line)
                continue

            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            char_vocab[char] = code
    print(char_vocab)


def read_code_vocab(filename, null_character=u'\u2591'):
    """
    return: {code: char}
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    code_vocab = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                if i == 0:
                    # code_vocab[0] = ' ' # no use
                    continue
                logging.warning('incorrect vocab file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            code_vocab[code] = char
    print(code_vocab)
    print(len(code_vocab))


def main():
    if flags.operation == 'build':
        build_vocab()
    elif flags.operation == 'read':
        filename = os.path.join(flags.input_dir, flags.name + '.txt')
        if flags.char_to_code:
            read_char_vocab(filename)
        else:
            read_code_vocab(filename)
    else:
        raise Exception


if __name__ == '__main__':
    main()
