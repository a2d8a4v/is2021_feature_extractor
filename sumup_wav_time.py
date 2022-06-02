import wave
import argparse
import contextlib

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default="data/train",
                    type=str)
args = parser.parse_args()

data_dir = args.data_dir

wavscp_dict = {}
with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]

text_dict = {}
levels    = set()
with open(data_dir + "/text", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        text_dict[info[0]] = info[1]
        levels.add(info[1])
nlevels_dict = {x:[] for x in list(levels)}

for utt_id, wav_path in wavscp_dict.items():
    fname = wav_path
    _level = text_dict[utt_id]
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        levels_dict[_level].append(duration)

for level_, data in levels_dict.items():
    print("Level-{}: {} seconds, {} utterances".format(level_, sum(data), len(data)))