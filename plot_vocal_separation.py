import io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
import audioop
import librosa.display

source_audio, sr = librosa.load('audio/rb-testspeech.mp3')

# compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(source_audio))

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)


##############################################
# The raw filter output can be used as a mask,
# but it sounds better if we use soft-masking.

# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
full = librosa.amplitude_to_db(S_full, ref=np.max)
librosa.display.specshow(full, y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
background = librosa.amplitude_to_db(S_background, ref=np.max)
librosa.display.specshow(background, y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()

plt.subplot(3, 1, 3)
foreground = librosa.amplitude_to_db(S_foreground, ref=np.max)
librosa.display.specshow(foreground, y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()

plt.tight_layout()
plt.show()

full_audio = librosa.istft(S_full)
foreground_audio = librosa.istft(S_foreground)
background_audio = librosa.istft(S_background)

####################################################
# Print out some metadata of the original audio and the 3 derived streams
print("sr: {}".format(sr))
print("orig({}) max {} power {}: {}".format(len(source_audio), audioop.max(source_audio,2), audioop.rms(source_audio,2), source_audio))
print("full({}) max {} power {}: {}".format(len(full_audio), audioop.max(background_audio,2), audioop.rms(full_audio,2), full_audio))
print("foreground({}) max {} power {}: {}".format(len(foreground_audio), audioop.max(background_audio,2), audioop.rms(foreground_audio,2), foreground_audio))
print("background({}) max {} power {}: {}".format(len(background_audio), audioop.max(background_audio,2), audioop.rms(background_audio,2), background_audio))

####################################################
# Write out the streams as sound files for later verification
audio_stream = io.BytesIO()
soundfile.write('orig.WAV', source_audio, sr)
#soundfile.write(audio_stream, source_audio, sr, format='WAV')
soundfile.write('full.WAV', full_audio, sr)
soundfile.write('fg.WAV', foreground_audio, sr)
soundfile.write('bg.WAV', background_audio, sr)
