# # from config import *
# # from openai import OpenAI
# # import os

# # def openai_generate_speech(audiofile, voice, text):
# # 	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # 	response = client.audio.speech.create(
# # 		model="tts-1",
# # 		voice=voice,
# # 		input=text
# # 	)
# # 	response.stream_to_file(audiofile)

# import os

# import torch
# import torchaudio
# import time
# from tortoise.api import TextToSpeech
# from tortoise.utils.audio import load_voices
# import humanize
# import datetime as dt

# def generate_speech(path_id, outfile, voice, text, speed="standard"):
#     tts = TextToSpeech(kv_cache=True, half=True)
#     selected_voices = voice.split(',')
#     for k, selected_voice in enumerate(selected_voices):
#         if '&' in selected_voice:
#             voice_sel = selected_voice.split('&')
#         else:
#             voice_sel = [selected_voice]
#         voice_samples, conditioning_latents = load_voices(voice_sel)

#         gen, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples, 
#                                              conditioning_latents=conditioning_latents,
#                                             return_deterministic_state=True,
#                                             preset=speed)
#         if isinstance(gen, list):
#             for j, g in enumerate(gen):
#                 torchaudio.save(os.path.join("temp", path_id, outfile), g.squeeze(0).cpu(), 24000)
#         else:
#             torchaudio.save(os.path.join("temp", path_id, outfile), gen.squeeze(0).cpu(), 24000)
 


# if __name__ == '__main__':
#     path_id = os.path.join("temp", "audio", str(int(time.time())))
#     os.makedirs(path_id, exist_ok=True)
#     tstart = time.time()
#     message = """Apple today confirmed that it will be permanently closing its Infinite Loop retail store in 
# Cupertino, California on January 20. Infinite Loop served as Apple's headquarters between the mid-1990s and 
# 2017, when its current Apple Park headquarters opened a few miles away."""
#     generate_speech(os.path.join("audio", str(int(time.time()))), "christmas.wav", "train_grace", 
#                     message, "ultra_fast")
        
#     # openai_generate_speech("speech.mp3", "onyx", 
#     #                 "Merry Christmas! May the holiday bring you endless joy, laughter, \
#     #                 and quality time with friends and family!")    
#     print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - tstart))))

import os
import torch
from TTS.api import TTS
import time
import humanize
import datetime as dt

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_speech(path_id, outfile, text, speaker_wav=None, language="en"):
    # Generate the full path for the output file
    output_path = os.path.join("temp",path_id, outfile)
    
    # Generate speech and save to a file
    tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=language)
    
    return output_path

def main():
    # Create a unique path based on the current timestamp
    path_id = os.path.join("temp", str(int(time.time())))
    os.makedirs(path_id, exist_ok=True)
    
    print(f"path_id: {path_id} path: {os.path.abspath(path_id)}")
    
    message = """Reading offers numerous benefits beyond simple entertainment, encompassing cognitive, emotional, and social dimensions. 
    It acts as a mental workout, sharpening critical thinking, concentration, and analytical skills while expanding vocabulary and knowledge. 
    Additionally, it fosters empathy and emotional intelligence by allowing readers to inhabit different perspectives and experiences. 
    Reading is a gateway to diverse cultures, histories, and ideas, promoting open-mindedness and cultural understanding. 
    Moreover, it provides relaxation and stress reduction, offering an escape from daily pressures. 
    Furthermore, it can enhance communication skills and creativity, sparking imagination and innovation. 
    Ultimately, reading is a lifelong pursuit that enriches individuals’ lives intellectually, emotionally, and socially."""

    speaker_wav = "/content/drive/MyDrive/Y2meta.app - Trump Makes CPAC Crowd Laugh Doing Mean Impression Of Biden Trying To Get Off Stage (256 kbps).mp3"
    outfile = "output.wav"
    language = "en"
    
    try:
        output_path = generate_speech(path_id, outfile, message, speaker_wav=speaker_wav, language=language)
        print(f"Speech saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    start = time.time()
    main()
    print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - start))))
