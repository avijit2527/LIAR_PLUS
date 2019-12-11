import pandas as pd
import numpy as np


#UserInfo.tsv
raw_data=pd.read_csv('dataset/test2.tsv',delimiter='\t',names=["sequence","id", "label", "statement", "subject","speaker","job_title",'state_info',"party","barely_true","false","half_true","mostly_true","pants_on_fire", "venue","justification"],encoding='utf-8')
data = raw_data[[ "label", "statement", "subject","job_title","party","justification"]].copy()
data = data.dropna().reset_index(drop=True)

def remove_special_characters(column):
    data[column] = data[column].str.replace('?', '')
    data[column] = data[column].str.replace(',', '')
    data[column] = data[column].str.replace('.', '')
    data[column] = data[column].str.replace('/', '')
    data[column] = data[column].str.replace(':', ' ')
    data[column] = data[column].str.replace(';', ' ')
    data[column] = data[column].str.replace('\"', '')
    data[column] = data[column].str.replace('\'', '')
    data[column] = data[column].str.replace('[', '')
    data[column] = data[column].str.replace(']', '')
    data[column] = data[column].str.replace('{', '')
    data[column] = data[column].str.replace('}', '')
    data[column] = data[column].str.replace('+', '')
    data[column] = data[column].str.replace('=', '')
    data[column] = data[column].str.replace('-', ' ')
    data[column] = data[column].str.replace('_', ' ')
    data[column] = data[column].str.replace(')', '')
    data[column] = data[column].str.replace('(', '')
    data[column] = data[column].str.replace('&', '')
    data[column] = data[column].str.replace('@', '')
    data[column] = data[column].str.replace('#', '')
    data[column] = data[column].str.replace('$', '')
    data[column] = data[column].str.replace('%', '')
    data[column] = data[column].str.replace('^', '')
    data[column] = data[column].str.replace('<', '')
    data[column] = data[column].str.replace('>', '')
    data[column] = data[column].str.replace('|', '')
    data[column] = data[column].str.replace('\\', '')
    data[column] = data[column].str.replace('"', '')
    data[column] = data[column].str.replace('*', '')
    data[column] = data[column].str.replace('~', '')
    data[column] = data[column].str.replace('!', '')
    data[column] = data[column].str.replace('`', '')
    data[column] = data[column].str.replace('0', '')
    data[column] = data[column].str.replace('1', '')
    data[column] = data[column].str.replace('2', '')
    data[column] = data[column].str.replace('3', '')
    data[column] = data[column].str.replace('4', '')
    data[column] = data[column].str.replace('5', '')
    data[column] = data[column].str.replace('6', '')
    data[column] = data[column].str.replace('7', '')
    data[column] = data[column].str.replace('8', '')
    data[column] = data[column].str.replace('9', '')
remove_special_characters("statement")
remove_special_characters("subject")
remove_special_characters("job_title")
remove_special_characters("party")
remove_special_characters("justification")


'''
#loading vordToVec pretrained
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


model = loadGloveModel("glove.6B/glove.6B.100d.txt")


def convert_word_to_vec(column):
    count_of_error = 0
    count_total_words = 0
    final_array = []
    for x in data[column].to_numpy(): 
        interim_array = []
        words = x.split(" ")
        for word in words:
            try:
                count_total_words = count_total_words +1
                temp = model[word.lower()]
                interim_array.append(temp)
            except KeyError:
                count_of_error = count_of_error + 1
                continue
        final_array.append(interim_array)
    np.save(column,final_array)
    print("Array Shape", np.array(final_array).shape)
    print("Error",count_of_error)  
    print("Total",count_total_words)




def save_labels():
    temp = []
    for x in data["label"].to_numpy(): 
        
        if x == 'pants-fire':
            temp.append(0)
        elif x == 'false':
            temp.append(1)
        elif x == 'barely-true':
            temp.append(2)
        elif x == 'half-true':
            temp.append(3)
        elif x == 'mostly-true':
            temp.append(4)
        elif x == 'true':
            temp.append(5)
   
            
    #print(len(temp))


save_labels()


convert_word_to_vec("statement")
convert_word_to_vec("subject")
convert_word_to_vec("job_title")
convert_word_to_vec("justification")

'''



data.to_csv(path_or_buf='dataset/test.tsv', sep='\t')













    
