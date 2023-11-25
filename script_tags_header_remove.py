#======================== Func: Text Cleanning
# it removes header (==Something Like This==) (== Or This ==) in raw text file
# and css tags that hard to remove in pyspark mapping, because the function that i declare below dk somehow shows that errors

## CSS Tags Example {wikitable: XXX....}

## that why i just make it to script file, after that do the afterwards preprocessing in jupyter notebook with pyspark

def removeHeadingsAndWeirdSpacings(sentence, wiki):
    clean_sentence = ''
    double_equals = 0
    curlyB = 0
    category_name = ''
    
    if wiki != "ms":
        category_name = 'Category:'
    else:
        category_name = 'Kategori:'
        
    for x in sentence.split(" "):
        if(double_equals >= 2):
            double_equals = 0
            
        if(curlyB >= 2):
            curlyB = 0
            
        if x.startswith("==") and len(x) > 2:
            continue
        
        if x.endswith("==") and len(x) > 2:
            continue
            
        if x.startswith("{"):
            curlyB += 1
            continue
        
        if x.endswith("}"):
            curlyB += 1
            continue
            
        if x == "==":
            double_equals += 1
            continue
            
        if x.startswith("Category:"):
            break
        
        if curlyB >= 1 and curlyB < 3:
            continue
        
        if double_equals < 1:
            clean_sentence += x + " "
            
    return clean_sentence

#======================== Func: Text Cleanning

#======================== Script: Text Cleanning

#{'1':'en', '2':'ms', '3':'zh'}
# NOTES
#1. the folder path must have 3 wiki files
#2. the input wiki text file name format is <number>parsed-extract-<name>wiki.txt
#3. following the dictionary in LINE 60 (The dictionary comment above *NOTES*), each dictionary element will show like {"number":"name"}
# Below is default dictionary, you can include or exclude to get different sets of cleaned text file(s)
#{'1':'en', '2':'ms', '3':'zh'}

# output file name format would be
# <name>-csv-test-remove-html.txt

wiki_names = {'2':'ms'}
for k,v in wiki_names.items():
    # text file name
    file_name = k + 'parsed-extract-'+ v +'wiki.txt'
    # default path that raw text file location, this follows the parsed data after runs Zi Xuan's parsed data download script
    folder_path = '/home/pc/assignment_final/data/'
    # access to parsed data folder
    file_path = folder_path + file_name
    num_lines = sum(1 for line in open(file_path))
    print("Processing " + file_name + " now. " + " Contains " + str(num_lines) + " lines 🙄")
    file  = open(file_path, 'r')
    tf_counter = 0
    line_counter = 0
    for line in file:
        # save the processed csv file save directory as script located
        with open(v+"-csv-test-remove-html.txt",'a') as file1:
                file1.write(removeHeadingsAndWeirdSpacings(line,v))
        line_counter += 1
        print(str(line_counter) + " out of " + str(num_lines) + "                 ", end="\r",flush=True)
    #loop finish
    print("TAGS and HEADER CLEANING "+ v +" WIKI TEXT FILE COMPLETE 👍")
    file.close()    
print("TAGS and HEADER CLEANING ALL WIKI FILES COMPLETE 👏\t👏")

#======================== Script: Text Cleanning