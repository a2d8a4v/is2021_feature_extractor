import os
import json
import pickle
import logging
import subprocess
import parselmouth
import numpy as np
from parselmouth.praat import run_file
import textgrid


class Error(Exception):
    pass

def pickleStore(savethings , filename):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return

def pikleOpen(filename):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p

def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)

def jsonSave(save_json, file_path):
    with open(file_path, "w") as output_file:
        json.dump(save_json, output_file, indent=4)

def opendict(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = l_[1:]
    return s

def opentext(file, col_start):
    s = set()
    with open(file, "r") as f:
        for l in f.readlines():
            for w in l.split()[col_start:]:
                s.add(w)
    return [w.lower() for w in list(s)]

def openappend(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            s[l.split()[0]] = l.split()[1]
    return s

def readwav(wav_path, rate):
    _, file_extension = os.path.splitext(wav_path)
    if file_extension.lower() == ".wav":
        import soundfile
        speech, rate = soundfile.read(wav_path)
    else:
        import numpy as np
        from pydub import AudioSegment
        speech = np.array(AudioSegment.from_mp3(wav_path).get_array_of_samples(), dtype=float)
    return speech, rate

def dict_miss_words(text, _dict):
    t = []
    for w in text.split():
        w = w.lower()
        if w not in _dict:
            t.append(w)            
    return t

def process_tltchool_gigaspeech_interregnum_tokens(tokens):
    disfluency_tltspeech  = [
        "@eh", "@ehm", "@em", "@mm", "@mmh", "@ns", "@nuh", "@ug", "@uh", "@um", "@whoah", "@unk"
    ]
    mapping = {
        "@ehm": "",
        "@mm": "",
        "@mmh": "",
        "@ns": "",
        "@nuh": "",
        "@ug": "",
        "@whoah": "",
        "@unk": "<unk>",
        "@uh": "UH",
        "@um": "UM",
        "@em": "EM",
        "@eh": "AH"
    }
    n = []
    for t in tokens.split():
        if t.lower() in disfluency_tltspeech:
            t = mapping[t.lower()]
        n.append(t)

    # BUG: not filter out the tokens if length is equal as one
    if not n:
        n = tokens.split()
    return " ".join(n)

def remove_gigaspeech_interregnum_tokens(tokens):
    disfluency_gigaspeech = ["AH", "UM", "UH", "EM", "ER", "ERR"]
    other_gigaspeech = ["<UNK>"]
    _gigaspeech = disfluency_gigaspeech+other_gigaspeech
    n = []
    for t in tokens.split():
        if t.upper() not in _gigaspeech:
            n.append(t)
    
    # BUG: not filter out the tokens if length is equal as one
    if not n:
        n = tokens.split()
    return " ".join(n)

def run_praat(file, *args, capture_output=True):

    assert os.path.isfile(file), "Wrong path to praat script"

    try:
        objects = run_file(file, *args, capture_output=capture_output)
        return objects
    except:
        print("Try again the sound of the audio was not clear")
        return None

def run_cmd(process, *args):
    """Run praat with `args` and return results as a c string
    Arguments:
    :param process: process name.
    :param *args: command line arguments to pass to praat.
    """
    p = subprocess.run([process] + list(map(str, list(args))), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)

    if p.returncode:
        raise Error(''.join(p.stderr.readlines()))
    else:
        return p.stdout

def from_textgrid_to_ctm(file_path):

    word_ctm_info = []
    phoneme_ctm_info = []

    tg = textgrid.TextGrid()
    tg.read(file_path)

    # Word-level
    words = tg.tiers[0]
    for i in range(len(words)):
        word = words[i].mark
        if word == "":
            continue
        start_time = round(words[i].minTime, 4)
        end_time = round(words[i].maxTime, 4)
        duration = round(end_time - start_time, 4)
        conf = round(1, 4)
        word_ctm_info.append(
            [word, start_time, duration, conf]
        )

    # Phoneme-level
    phonemes = tg.tiers[0]
    for i in range(len(phonemes)):
        phoneme = phonemes[i].mark
        if phoneme == "":
            continue
        start_time = round(phonemes[i].minTime, 4)
        end_time = round(phonemes[i].maxTime, 4)
        duration = round(end_time - start_time, 4)
        conf = round(1, 4)
        phoneme_ctm_info.append(
            [phoneme, start_time, duration, conf]
        )

    return word_ctm_info, phoneme_ctm_info

def fix_data_type(data_dict):
    rtn = {}
    for term, data in data_dict.items():
        if isinstance(data, np.float32):
            data = np.float64(data)
        if isinstance(data, dict):
            data = fix_data_type(data)
        if isinstance(data, np.ndarray):
            data = data.tolist()
        rtn[term] = data
    return rtn

def get_from_dict(_dict, keys_list):

    rtn = {}
    for key in keys_list:
        rtn[key] = _dict.get(key)
    return rtn
        
