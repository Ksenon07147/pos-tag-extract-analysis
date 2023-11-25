In ipynb subfolder, there will be 5 folders and 1 python script.

In these 4 folders(Social_Media, wiki_en, wiki_ms and wiki_zh), each folder contains preprocessing, keyword extract and pos tagging Jupyter Notebook.

And for HBase_Store_Retrieve folder, inside there's only one Jupyter Notebook about Python Codes for Storing and Retrieving POS-Tagging result into HBase.

And the one script is the script that performing the very first preprocessing for the raw text files before imported into Jupyter Notebook for preprocessing.
-- In this script there are already commented for every step performed and things to notice.
-- In default, we assume that raw text file is in /home/pc/data/parsed_data/
-- The script didn't include social-media raw text file (should be CSV for social media) because it can directly do preprocessing without running this script.
-- Please take a look of python script because there may need have some edit, like choosing raw text files that you guys want to do processing
-- The final product of this script, depends on terminal's pointer location (Example if script runs when pointer locate in ~ path (Home Path), the processed raw text file will also locate in ~ path)