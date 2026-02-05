import os 
import queue
import natsort
import json
import threading
import glob
import regex as re
import ast
import pandas as pd
import glob
max_threads = 5

multithread_semaphore = threading.Semaphore(value = max_threads)

# gets the text from a bunch of text files in the in_dir argument it will only take .txt files 
def get_text_from_dir(in_dir: str,min_text_length:int=150000 ,verbose: bool = False) -> list[str]:
    out_text = []
    if(verbose):
        print(f"getting files from dir: {in_dir}")
    files = natsort.natsorted(os.listdir(in_dir))
    for file in files:
        if(os.path.isdir(file)):
            files.remove(file)
        if(os.path.splitext(file)[1] != ".txt"):
            files.remove(file)
    for file in files:
        curr_dict = {}
        curr_dict["index"] = int(re.search("(\d+)",file).group(0))
        with open(in_dir+"\\"+file,"r",encoding="utf-8")as outfile:
            text = outfile.read()
            if(len(text)<min_text_length):
                continue
            curr_dict["text"] = text
        out_text.append(curr_dict)
    return out_text,curr_dict


# gets the entropy information from the json files that nero outputs
def get_entropies_from_dir(in_dir: str,verbose:bool=False):
    out_list = []
    files = natsort.natsorted(os.listdir(in_dir))
    if(verbose):
        print(f"found files in dir {files}")
    for file in files:
        if(os.path.isdir(file)):
            if(verbose):
                print(f"removing dir {file}")
            files.remove(file)
        if(os.path.splitext(file)[1] != ".json"):
            if(verbose):
                print(f"removing non json file {file}")
                print(f"actual extension {os.path.splitext(file)[1]}")
            files.remove(file)
    for file in files:
        curr_dict = {"entropy":0.0,
                     "entropy_list":[],
                     "index": 0}
        with open(in_dir+"\\"+file,"r") as infile:
            data = json.load(infile)
            curr_dict["entropy"] = data["average_entropy_rate"]
            curr_dict["entropy_list"] = data["entropy_rate"]
            curr_dict["index"] = int(re.search("(\d+)",file).group(0))
            out_list.append(curr_dict)
    return out_list

#wrapper for the get novel function that puts the function in its own thread and starts it
# spawns one thread that waits until the function is finished and releases the semaphore that the thread acquired so that calling function isnt blocked by join   
def multithreading(func):
    def wrapper(*args,**kwargs):
        multithread_semaphore.acquire()
        print(F"multithread threding count: {multithread_semaphore._value}")
        t = threading.Thread(target=func,args=args,kwargs=kwargs)
        t.start()
        wait = threading.Thread(target=thread_end_wait,args=(t,multithread_semaphore))
        wait.start()
    return wrapper


def thread_end_wait(t,sem):
    t.join()
    sem.release()

def get_all_files_with_extension(in_dir,extension):
    return glob.glob(f"*{extension}", root_dir=in_dir)


def get_obj_col_lst(df: pd.DataFrame,col: str,verbose:bool = False):
    lst = [ast.literal_eval(l) for l in df[col].to_list()]
    if(verbose):
        print(lst)
    return lst

def table_exclude_length(df: pd.DataFrame,folder:str,length:int):
    for row in df.iterrows():
        file = glob.glob(f"*{row.index}*.txt",root_dir=folder)
        if(file):
            with open(file,"r",encoding="utf-8") as infile:
                text = infile.read()
            if(len(text)<length):
                df.drop(row.index)

def collapse_subjects(subject):
    """
    Map a Project Gutenberg (or similar) 'Subjects' string to one of nine
    high-level categories required for the entropy-rate study.

    Categories (returned values)
    ----------------------------
    • Fiction
    • Children/Youth
    • Poetry
    • Drama
    • History
    • Religion
    • Reference
    • Science/Philosophy
    • Other Non-Fiction   (catch-all; should now be very small)
    """
    s = str(subject).lower()

    # --- 1. Children / Youth -------------------------------------------------
    if any(k in s for k in
           ('juvenile', 'children', 'fairy tales', 'folklore', 'nursery', 'youth')):
        return 'Children/Youth'

    # --- 2. Poetry ------------------------------------------------------------
    if 'poetry' in s:
        return 'Poetry'

    # --- 3. Drama -------------------------------------------------------------
    if any(k in s for k in ('drama', 'play', 'libretto', 'librettos', 'theatre', 'theater')):
        return 'Drama'

    # --- 4. Fiction -----------------------------------------------------------
    FICTION_KEYS = (
        'fiction', 'story', 'stories', 'adventure', 'detective', 'mystery', 'spy ',
        'horror', 'fantasy', 'satire', 'humorous', 'utopia', 'dystopia',
        'bildungsroman', 'western', 'allegor', 'christmas stories', 'college stories',
        'black humor', 'imaginary letters', 'anecdotes')  # plus plural/variants
    if any(k in s for k in FICTION_KEYS):
        return 'Fiction'

    # --- 5. History -----------------------------------------------------------
    HISTORY_KEYS = (
        'history', 'historical', 'world war', 'civilization', 'colonial period',
        'monarchy', 'dynasty', 'middle ages', 'renaissance', 'early modern',
        'biography', 'contemporaries', 'massachusetts', 'italy', 'japan',
        'soviet union', 'social conditions', 'politics and government',
        'politics', 'communism', 'social problems', 'achilles', 'chaucer','civil disobedience', 'southwest',
        'utopias', 'sources', 'travel', 'description and travel')
    if any(k in s for k in HISTORY_KEYS):
        return 'History'

    # --- 6. Religion ----------------------------------------------------------
    RELIGION_KEYS = (
        'religion', 'bible', 'sacred', 'doctrines', 'theology',
        'church', 'yoga', 'hymn', 'prayer', 'sermon')
    if any(k in s for k in RELIGION_KEYS):
        return 'Religion'

    # --- 7. Reference ---------------------------------------------------------
    REFERENCE_KEYS = (
        'reference', 'language', 'grammar', 'synonyms', 'dictionary', 'manual',
        'handbook', 'guide', 'quotations', 'essays', 'english essays',
        'american essays', 'translations into english', 'classic literature',
        'classical literature', 'adaptations', 'book collecting', 'bibliomania')
    if any(k in s for k in REFERENCE_KEYS):
        return 'Reference'

    # --- 8. Science / Philosophy ---------------------------------------------
    SCI_KEYS = (
        'science', 'scientific', 'math', 'physics', 'biology', 'chemistry',
        'evolution', 'darwin', 'dimension', 'technology', 'computers',
        'methodology', 'plant', 'plants', 'spiders', 'earthworms', 'climbing')
    PHILOSOPHY_KEYS = ('philosophy', 'logic', 'ethics', 'metaphysics')
    if any(k in s for k in SCI_KEYS) or any(k in s for k in PHILOSOPHY_KEYS):
        return 'Science/Philosophy'

    # ---



    # --- 9. Other Non-Fiction -------------------------------------------------
    return 'Other Non-Fiction'


def collapse_list_subjects(lst):
    for index,subject in enumerate(lst):
        lst[index] = collapse_subjects(subject)

    return lst

def get_json_data(d):
    files = glob.glob("*.json",root_dir=d)
    for file in files:
        path = f"{d}\\{file}"
        with open(path,"r") as infile:
            yield (path,json.load(infile))
