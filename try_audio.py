import numpy as np
import librosa
from librosa import display
import soundfile
import matplotlib.pyplot as plt

y, sr = librosa.load('audio/rb-testspeech.mp3', duration=5)
S_full, phase = librosa.magphase(librosa.stft(y))

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 2, 10
power = 2

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

S_foreground = mask_v * S_full

full = librosa.amplitude_to_db(S_full, ref=np.max)
librosa.display.specshow(full, y_axis='log', sr=sr)

plt.title('Full spectrum')
plt.colorbar()

plt.tight_layout()
plt.show()

print("y({}): {}".format(len(y),y))
print("sr: {}".format(sr))

full_audio = librosa.istft(S_full)
foreground_audio = librosa.istft(S_foreground)
print("full({}): {}".format(len(full_audio), full_audio))

soundfile.write('orig.WAV', y, sr)
soundfile.write('full.WAV', full_audio, sr)
soundfile.write('foreground.WAV', foreground_audio, sr)
