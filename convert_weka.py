import os
from os import listdir
from os.path import isfile, join

def write_to_file(filename, text, append=True, add_linefeed=True):
    if append is True:
        mode = 'a'
    else:
        mode = 'w'
    linefeed = ''
    if add_linefeed is True:
        linefeed = '\n'
    with open(filename, mode) as fw:
        fw.write(str(text) + linefeed)

def write_to_file_buffered(filename, text_list, append=True, buffer_size=10000):
    print('Writing into: ' + filename)
    counter = 0
    temp_str = ""
    for text in text_list:
        if counter <= buffer_size:
            temp_str = temp_str + text + '\n'
        else:
            write_to_file(filename, temp_str, append, add_linefeed=False)
            temp_str = ""
            counter = 0
        counter += 1
    # Write remaining text
    if temp_str != "":
        write_to_file(filename, temp_str, append, add_linefeed=False)

def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
        return True
    except OSError as exception:
        return False

files = []
folder = './'
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

for file in onlyfiles:
    if '.csv' in file:
        files.append(file)

print(files)
for file in files:
    print(file)
    output_file = file.split('.')[0] + '.arff'
    print(output_file)
    header = []
    texts = []
    categories = {}
    n_col = 0
    with open(file, 'r') as fr:
        for line in fr:
            split = line.strip().split(',')
            n_col = len(split)
            for i in range(len(split)):
                try:
                    if split[i] == '?' or split[i] == '':
                        pass
                    else:
                        x = float(split[i])
                except:
                    cat = categories.get(i, None)
                    if cat is None:
                        cat = []
                    if split[i] not in cat:
                        cat.append(split[i])
                    categories[i] = cat
            texts.append(line.strip())
        ### Construct the header
        header.append('@Relation FVS')
        for i in range(n_col):
            att_name = 'ATT_{}'.format(i)
            cat = categories.get(i, None)
            if cat == None:
                att_type = 'NUMERIC'
            else:
                att_type = '{' + ','.join(cat) + '}'
            header.append('@attribute {}\t{}'.format(att_name, att_type))
        header.append('@data')
        remove_file_if_exists(output_file)
        write_to_file_buffered(output_file, header)
        write_to_file_buffered(output_file, texts)