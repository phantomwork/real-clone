from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch

device = 'cpu' # or 'cpu'
model = load_codec_model(use_gpu=True if device == 'cuda' else False)

# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
from hubert.hubert_manager import HuBERTManager
hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
# Load HuBERT for semantic tokens
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

# Load the CustomTokenizer model
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)  # Automatically uses the right layers

# Load and pre-process the audio waveform
audio_filepath = 'audio.wav' # the audio you want to clone (under 13 seconds)
wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.to(device)


semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)

semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)
#%%
# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
#%%
# move codes to cpu
codes = codes.cpu().numpy()
# move semantic tokens to cpu
semantic_tokens = semantic_tokens.cpu().numpy()
#%%
import numpy as np
voice_name = 'output' # whatever you want the name of the voice to be
output_path = 'bark/assets/prompts/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
#%%
# That's it! Now you can head over to the generate.ipynb and use your voice_name for the 'history_prompt'
#%%

#%%
# Heres the generation stuff copy-pasted for convenience
#%%
from bark.api import generate_audio
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

# Enter your prompt and speaker here
text_prompt = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
voice_name = "output" # use your custom voice name here if you have one
#%%
# download and load all models
preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
    path="models"
)
#%%
# simple generation
audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
#%%
# generation with more control
x_semantic = generate_text_semantic(
    text_prompt,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)

x_coarse_gen = generate_coarse(
    x_semantic,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)
x_fine_gen = generate_fine(
    x_coarse_gen,
    history_prompt=voice_name,
    temp=0.5,
)
audio_array = codec_decode(x_fine_gen)
#%%
from IPython.display import Audio
# play audio
Audio(audio_array, rate=SAMPLE_RATE)
#%%
from scipy.io.wavfile import write as write_wav
# save audio
filepath = "/output/audio.wav" # change this to your desired output path
