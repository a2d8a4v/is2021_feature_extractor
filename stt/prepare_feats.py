import os
import json
import argparse
import logging
from tqdm import tqdm
import numpy as np
from espnet_models import SpeechModel
from audio_models import AudioModel
from vad_model import VadModel
from g2p_model import G2PModel
from espnet.utils.cli_utils import strtobool
from utils import (
    pickleStore,
    pikleOpen,
    jsonLoad,
    opendict,
    opentext,
    readwav,
    fix_data_type,
    run_cmd,
    get_from_dict
)

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

"""
:Gigaspeech: Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave
:Librispeech: espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp
"""
parser.add_argument("--model_tag",
                    default="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave", #https://github.com/espnet/espnet/issues/3018: LSTM pretrained-model
                    type=str)

parser.add_argument("--text_path",
                    default="data/train/text",
                    type=str)

parser.add_argument("--model_name",
                    default="gigaspeech",
                    type=str)

parser.add_argument("--ngpu",
                    default=0,
                    type=str)

parser.add_argument("--s2t",
                    default=True,
                    type=strtobool)

parser.add_argument("--sample_rate",
                    default=16000,
                    type=int)

parser.add_argument("--vad_mode",
                    default=1,
                    type=int)

parser.add_argument("--tag",
                    default="",
                    type=str)

# We accept both ESPNet-based and Kaldi-based lexicon
parser.add_argument("--dict",
                    default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/acoustic_phonetic_features/data/local/dict/lexicon.txt",
                    type=str,
                    required=True)

parser.add_argument("--long_decode_mode",
                    default=False,
                    type=strtobool)

parser.add_argument("--ctc_aligner",
                    default=True,
                    type=strtobool)

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
output_dir = os.path.join(data_dir, model_name+"_"+args.tag)
tmp_apl_decoding = "tmp_apl_decoding_"+args.tag if args.tag else "tmp_apl_decoding"

# Temporary data saved for ToBI
tobi_path = os.path.join(data_dir, tmp_apl_decoding, "tobi")
if not os.path.isdir(tobi_path):
    os.makedirs(tobi_path)

# logging file path assign
logging.basicConfig(filename=data_dir + "/prepare_feats_log", level=logging.DEBUG)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

sample_rate = args.sample_rate
vad_mode = args.vad_mode
tag = args.model_tag
ctc_aligner = args.ctc_aligner
wavscp_dict = {}
text_dict = {}
utt_list = []
err_list = []
all_info = {}
ctm_dict = {} # stt and ctm
word2phn_dict = opendict(args.dict) # word to phone dict

# validate the dict has all the words inside
_t = opentext(args.text_path, 1)
_w = sum([ 1 for w in _t if w.lower() not in list(word2phn_dict.keys())])
assert _w == 0, "{} words are not inside the dictionary!".format(_w)

# move text load beforehand to avoid unpredicable error in dict and save more time and your GPU memory
with open(args.text_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:]).upper() # Due to the tokens inside gigaspeech ad wsj is upper-case, we need to change the text to upper-case also.

device = "cuda" if int(args.ngpu) > 0 else "cpu"

speech_model = SpeechModel(tag, device)
audio_model = AudioModel(sample_rate)
vad_model = VadModel(vad_mode, sample_rate)
g2p_model = G2PModel()

with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

if os.path.isfile(output_dir + "/error"):
    with open(output_dir + "/error", "r") as fn:
        for utt_id in fn.readlines():
            err_list.append(utt_id.split()[0])

if ctc_aligner:
    print('ToBI information can not be retrieved due to phoneme issue in End-to-End model')

if args.s2t:
    print("Free Speaking")
else:
    print("Reading")

## Due to the memory problem if load all data into memory
if args.long_decode_mode:
    if os.path.exists(os.path.join(data_dir, tmp_apl_decoding+".list")):
        print("Decoded features loading...")
        with open(os.path.join(data_dir, tmp_apl_decoding+".list"), "r") as fn:
            for l in tqdm(fn.readlines()):
                l_ = l.split()
                # We do not need to keep too much data in memory during inference, we can just load all the data after end the inference
                all_info[l_[0]] = {} if args.long_decode_mode else pikleOpen(l_[1])

print("Decoding Start")

for utt_id in tqdm(utt_list):

    if utt_id in all_info:
        continue

    # skip error
    if utt_id in err_list:
        continue

    wav_path = wavscp_dict[utt_id]
    text_prompt = text_dict[utt_id]

    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    audio, rate = vad_model.read_wave(wav_path)

    # @https://stackoverflow.com/questions/56436975/what-is-2-15-doing-in-this-line-of-code
    speech = np.frombuffer(audio, dtype='int16').astype(np.float32) / 32768.0
    assert rate == 16000

    total_duration = speech.shape[0] / rate

    # audio feature
    f0_list, f0_info = audio_model.get_f0(speech)
    energy_list, energy_info = audio_model.get_energy(speech)
    formants_info = audio_model.get_formants(wav_path, speech, rate)

    # We tested to use s2t mode before, and we found that the probabilities of the word predicted by model is at a normal range. Thus, we consider using prompt as input but get confidence score 0 is also a normal thing, not the problem with the acouostic model trained by gigaspeech.
    if args.s2t: # for free speaking condition

        # fluency feature and confidence feature
        # BUG: Some audio are too long, need to partitate by VAD before recognizing
        speechs = vad_model.get_speech_segments(audio, rate)
        text = []
        for speech_seg in speechs:
            text_seg = speech_model.recog(speech_seg)
            text.append(text_seg)
        text = " ".join(" ".join(text).split())

        # BUG: Because of the two dictionaries have different words, we need to extend the dictionary with G2P toolkit
        word2phn_dict = g2p_model.g2p(text, word2phn_dict)
    else:
        text = text_dict[utt_id]
    
    # alignment (stt)
    # NOTICE: the tatal words in text and ctm should be the same
    if ctc_aligner:
        ctm_info = speech_model.get_ctm(speech, text)
    else:
        textgrid_file_path = speech_model.get_textgrid_mfa(text, wav_path, os.path.join(data_dir, tmp_apl_decoding, 'lexicon'), utt_id, word2phn_dict=word2phn_dict)
        ctm_info = speech_model.get_ctm_from_textgrid(textgrid_file_path)

    phone_ctm_info, phone_text = speech_model.get_phone_ctm(ctm_info, word2phn_dict)

    sil_feats_info, response_duration = speech_model.sil_feats(ctm_info, total_duration)
    word_feats_info, response_duration = speech_model.word_feats(ctm_info, total_duration)
    phone_feats_info, response_duration = speech_model.phone_feats(phone_ctm_info, total_duration)

    rhythm_feats_info = speech_model.rhythm_feats(ctm_info, word2phn_dict)
    pitch_feats_info = audio_model.get_pitch(speech, ctm_info, f0_list)
    intensity_feats_info = audio_model.get_intensity(speech, ctm_info, energy_list)

    # NOTICE: ASR decoding should be based on HMM-DNN ASR model with Librispeech, or just use mfa toolkit)
    if not ctc_aligner:
        # BUG: need to use absolute path here
        tobi_feats_info = audio_model.get_tobi(os.path.abspath(wav_path), os.path.abspath(tobi_path), os.path.abspath(textgrid_file_path))

    # Save data
    save = { "stt": text, "stt(g2p)": phone_text, "prompt": text_prompt,
                        "wav_path": wav_path, "ctm": ctm_info, 
                        "feats": {  **f0_info, **energy_info, 
                                    **sil_feats_info, **word_feats_info,
                                    **phone_feats_info,
                                    "pitch": pitch_feats_info,
                                    "intensity": intensity_feats_info,
                                    "formant": formants_info,
                                    "rhythm": rhythm_feats_info,
                                    "total_duration": total_duration,
                                    "response_duration": response_duration}
           }

    # Save data for ToBI
    if not ctc_aligner:
        save['feats']['tobi'] = tobi_feats_info

    if not args.long_decode_mode:
        all_info[utt_id] = fix_data_type(save)

    # due to spending too much time on decoding, we need to save each decoding result per utt
    if args.long_decode_mode:
        fp = os.path.join(data_dir, tmp_apl_decoding)
        if not os.path.isdir(fp):
            os.mkdir(fp)
        pickleStore(save, os.path.join(fp, utt_id+".pkl"))
        with open(os.path.join(data_dir, tmp_apl_decoding+".list"), "a") as fn:
            fn.write("{} {}\n".format(utt_id, os.path.join(fp, utt_id+".pkl")))

# For Saving all data, import data to all_info
if args.long_decode_mode:
    print("Decoded features Saving...")
    with open(os.path.join(data_dir, tmp_apl_decoding+".list"), "r") as fn:
        for l in tqdm(fn.readlines()):
            l_ = l.split()
            all_info[l_[0]] = fix_data_type( pikleOpen(l_[1]) )

# Print out the output dir
print(output_dir)

# record error
if err_list:
    with open(output_dir + "/error", "w") as fn:
        for utt_id in err_list:
            fn.write(utt_id + "\n")

with open(output_dir + "/all.json", "w") as fn:
    json.dump({"utts": all_info}, fn, indent=4)

# write STT Result to file
with open(output_dir + "/text", "w") as fn:
    for utt_id in utt_list:

        # skip error
        if utt_id in err_list:
            continue

        fn.write(utt_id + " " + all_info[utt_id]["stt"] + "\n")

# write alignment results fo file
with open(output_dir + "/ctm", "w") as fn:
    end_time = -100000
    for utt_id in utt_list:

        # skip error
        if utt_id in err_list:
            continue

        ctm_infos = all_info[utt_id]["ctm"]
        for i in range(len(ctm_infos)):
            text_info, start_time, duration, conf = ctm_infos[i]
            # utt_id channel start_time duration text conf
            ctm_info = " ".join([utt_id, "1", str(start_time), str(duration), text_info, str(conf)])
            fn.write(ctm_info + "\n")
