import pandas as pd
import numpy as np



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


def set_binary_dataset():
    data["label"] = data["label"].str.replace('false', 'labelZero')  
    data["label"] = data["label"].str.replace('barely-true', 'labelZero')    
    data["label"] = data["label"].str.replace('pants-fire', 'labelZero')
    data["label"] = data["label"].str.replace('mostly-true', 'labelOne')
    data["label"] = data["label"].str.replace('half-true', 'labelOne') 
    data["label"] = data["label"].str.replace('true', 'labelOne')   







#cleaning and Saving TEST data
raw_data=pd.read_csv('dataset/test2.tsv',delimiter='\t',names=["sequence","id", "label", "statement", "subject","speaker","job_title",'state_info',"party","barely_true","false","half_true","mostly_true","pants_on_fire", "venue","justification"],encoding='utf-8')
data = raw_data[[ "label", "statement", "subject","job_title","party","justification"]].copy()
data = data.dropna().reset_index(drop=True)

remove_special_characters("statement")
remove_special_characters("subject")
remove_special_characters("job_title")
remove_special_characters("party")
remove_special_characters("justification")

data.to_csv(path_or_buf='dataset/test.tsv', sep='\t')

set_binary_dataset()
data.to_csv(path_or_buf='dataset/test_binary.tsv', sep='\t')



#cleaning and Saving TRAIN data
raw_data=pd.read_csv('dataset/train2.tsv',delimiter='\t',names=["sequence","id", "label", "statement", "subject","speaker","job_title",'state_info',"party","barely_true","false","half_true","mostly_true","pants_on_fire", "venue","justification"],encoding='utf-8')
data = raw_data[[ "label", "statement", "subject","job_title","party","justification"]].copy()
data = data.dropna().reset_index(drop=True)

remove_special_characters("statement")
remove_special_characters("subject")
remove_special_characters("job_title")
remove_special_characters("party")
remove_special_characters("justification")


data.to_csv(path_or_buf='dataset/train.tsv', sep='\t')

set_binary_dataset()
data.to_csv(path_or_buf='dataset/train_binary.tsv', sep='\t')



#cleaning and Saving VALIDATION data
raw_data=pd.read_csv('dataset/val2.tsv',delimiter='\t',names=["sequence","id", "label", "statement", "subject","speaker","job_title",'state_info',"party","barely_true","false","half_true","mostly_true","pants_on_fire", "venue","justification"],encoding='utf-8')
data = raw_data[[ "label", "statement", "subject","job_title","party","justification"]].copy()
data = data.dropna().reset_index(drop=True)

remove_special_characters("statement")
remove_special_characters("subject")
remove_special_characters("job_title")
remove_special_characters("party")
remove_special_characters("justification")


data.to_csv(path_or_buf='dataset/val.tsv', sep='\t')

set_binary_dataset()
data.to_csv(path_or_buf='dataset/val_binary.tsv', sep='\t')


    
