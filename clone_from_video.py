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
audio_filepath = 'G:/Projects/real-clone/output.wav' # the audio you want to clone (under 13 seconds)
torchaudio.set_audio_backend("soundfile")
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
text_prompt = "Hi there, everyone! Today we are going to talk about why you should think twice before eating burgers. Burgers might seem yummy and fun to eat, but they can be really bad for your health. Let's find out why..First, burgers are often made with lots of fatty meats. These fats, called saturated fats, can make your heart unhealthy. When you eat too many burgers, you can start to have problems with your heart. It’s like putting bad fuel into a car—the car won't run well and it might even break down. Your heart is the same way; it needs good fuel, like fruits and veggies, to work properly..Second, burgers usually come with lots of extra stuff that isn't good for you. Think about the cheese, bacon, and sauces that are often added. These toppings can have a lot of salt. Eating too much salt can make your blood pressure go up, which is not good for your body. It's like adding too much air to a balloon; one day, it might just pop..Another thing to consider is that burgers can make you gain weight. They usually have a lot of calories because of the fatty meat and other ingredients. Eating too many calories can add extra weight to your body, making it harder to run and play. This can also lead to other serious health issues like diabetes..It’s also important to remember that many burgers are made quickly and might not be fresh. They can have harmful bacteria that make you sick. You don’t want to end up spending all your fun time in bed with a stomach ache!.Instead of burgers, try eating foods that are good for you and taste great too. Make a homemade sandwich with fresh veggies, lean meat, and whole grain bread. Your body will thank you for it!.So, next time you think about grabbing a burger, remember these tips and make a better choice for your health. Stay happy and stay healthy!"
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
write_wav(filepath, SAMPLE_RATE, audio_array)