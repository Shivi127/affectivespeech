import sys
import logging
from queue import Queue
from queue import Empty
import io
import time
from datetime import datetime
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
import audioop
import librosa.display

from background_process import Background
from show_text_fe import show_text
from sound_state import *
from plotutil import *

VOLUME_SILENCE_RANGE = 1.10  # Consider anything within 10% above the minimum sound to be background noise
VOLUME_RAISED_VARIANCE_THRESHOLD = 0.5
VOLUME_LOWERED_VARIANCE_THRESHOLD = -1 * VOLUME_RAISED_VARIANCE_THRESHOLD

STATE_SAMPLE_LIFETIME_SECS = 5
PAUSE_MINIMUM_SPAN_SECS = 2

def get_max(audio_chunk):
    return audioop.max(audio_chunk, 2)

def get_rms(audio_chunk):
    return audioop.rms(audio_chunk, 2)

def get_avg(audio_chunk):
    return audioop.avg(audio_chunk, 2)

def convert_audio_data(sound_bite):
    logging.debug(type(sound_bite))
    logging.debug(len(sound_bite))
    audio_data = np.frombuffer(sound_bite, dtype=np.int16).astype(np.float32)
    logging.debug(type(audio_data))
    logging.debug(len(audio_data))
    logging.debug(type(audio_data[0]))
    return audio_data

def extract_audio_spectrograms(audio_data, sampling_rate):
    # Compute the spectrogram magnitude and phase
    spectrogram_full, phase = librosa.magphase(librosa.stft(audio_data))

    spectrogram_filter = librosa.decompose.nn_filter(spectrogram_full,
        aggregate=np.median, metric='cosine',
        width=int(librosa.time_to_frames(2, sr=sampling_rate)))

    spectrogram_filter = np.minimum(spectrogram_full, spectrogram_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(spectrogram_filter, 
        margin_i * (spectrogram_full - spectrogram_filter), power=power)

    mask_v = librosa.util.softmask(spectrogram_full - spectrogram_filter,
         margin_v * spectrogram_filter, power=power)

    spectrogram_foreground = mask_v * spectrogram_full
    spectrogram_background = mask_i * spectrogram_full

    return (spectrogram_full, spectrogram_background, spectrogram_foreground)
 
class SoundConsumer(Background):
    def run(self):
        self._initLogging()
        logging.debug("run")
        self.consume_raw_audio()

    def add_to_recent_window_rms(self, window_rms):
        self._sound_windows.append(window_rms)

    def add_to_recent_samples(self, sample):
        self._sound_samples.append(sample)

    def calculate_function_average_above_threshold_for_recent_samples(self, function, threshold):
        slice_start = -1 * min(self.plot_sample_count, len(self._sound_samples))
        applied_function_results = [eligible_sample for eligible_sample in [function(sample) for sample in list(self._sound_samples)[slice_start:] if sample is not None] if eligible_sample > threshold]
        if applied_function_results is None or not applied_function_results:
            return None
        return sum(applied_function_results) / len(applied_function_results)

    def calculate_function_average_for_samples(self, function):
        applied_function_results = [function(sample) for sample in self._sound_samples if sample is not None]
        if applied_function_results is None or not applied_function_results:
            return None
        return sum(applied_function_results) / len(applied_function_results)

    def calculate_function_min_for_samples(self, function):
        applied_function_results = [function(sample) for sample in self._sound_samples if sample is not None]
        if applied_function_results is None or not applied_function_results:
            return None
        logging.debug('min of {}'.format(applied_function_results))
        return min(applied_function_results)

    def __init__(self, audio_chunk_pipe, sampling_rate, log_queue, log_level, plot_sample_count):
        super(SoundConsumer, self).__init__(audio_chunk_pipe, log_queue, log_level)
        self.current_state = STATE_VOLUME_CONSTANT
        self._sound_samples = deque()
        self._sound_windows = deque()
        self._state_changes = deque()
        self.current_pause_start = None
        self.current_pause_end = None
        self.plot_sample_count = plot_sample_count
        self.sampling_rate = sampling_rate
        self.all_min = 9999
        self.all_max = -9999

    def maintain_state_change_queue_lifetime(self):
        """
        Guarantee that only state changes that occured in the desired lifetime are kept.
        returns: If any stored entries were old and deleted.
        """
        oldest_target_sample_age = time.time() - STATE_SAMPLE_LIFETIME_SECS
        deleted = False
        
        while len(self._state_changes) > 0 and self._state_changes[0][2] < oldest_target_sample_age:
            self._state_changes.popleft()
            deleted = True
        return deleted

    def record_state_change(self, state, change_sample_start_at, change_sample_end_at):
        self.current_state = state
        sample = (state, change_sample_start_at, change_sample_end_at)
        self._state_changes.append(sample)
        self.maintain_state_change_queue_lifetime()

    def _truncate_recent_samples(self):
        self._sound_samples = deque()
        self._sound_windows = deque()

    def plot_recent_samples_rms(self):
        slice_start = -1 * min(self.plot_sample_count, len(self._sound_samples))
        samples_plot = [get_rms(sample) for sample in list(self._sound_samples)[slice_start:]]
        windows_plot = [window for window in list(self._sound_windows)[slice_start:]]
        logging.debug('plot: {} {}'.format(samples_plot, windows_plot))
        draw_graph('sound level', (self.all_min, self.all_max), 'RMS', samples_plot, windows_plot)

    def consume_raw_audio(self):
        logging.info("Waiting to consume audio")
        while not self._exit.is_set():
            try:
                logging.debug("recv")
                seq, chunk_size, start_at, end_at, sound_bite = self._receive_pipe.recv()
                if sound_bite is None:
                    logging.debug('NULL audio chunk')
                audio_data = convert_audio_data(sound_bite)
                full_spectrogram, background_spectrogram, foreground_spectogram = extract_audio_spectrograms(audio_data, self.sampling_rate)
                sample_rms = get_rms(sound_bite)
                self.all_min = min(sample_rms, self.all_min)
                self.all_max = max(sample_rms, self.all_max)
                window_min = self.calculate_function_min_for_samples(get_rms)
                if window_min is None:
                    window_min = sample_rms
                volume_silence_threshold = window_min * VOLUME_SILENCE_RANGE
                self.add_to_recent_samples(sound_bite)
                if sample_rms > volume_silence_threshold:
                    self.current_pause_start = None
                    self.current_pause_end = None
                else:
                    if self.current_pause_start is None:
                        self.current_pause_start = start_at
                    self.current_pause_end = end_at
                    if self.current_pause_end - self.current_pause_start >= PAUSE_MINIMUM_SPAN_SECS:
                        logging.debug('discard {} samples\n'.format(len(self._sound_samples)))
                        self._truncate_recent_samples()
 
                window_rms = self.calculate_function_average_above_threshold_for_recent_samples(get_rms, volume_silence_threshold)
                if window_rms is None:
                    window_rms = sample_rms
                self.add_to_recent_window_rms(window_rms)
                sample_rms_delta = sample_rms - window_rms

                sample_rms_variance = float(sample_rms_delta) / window_rms

                if sample_rms_variance >= VOLUME_RAISED_VARIANCE_THRESHOLD:
                    self.record_state_change(STATE_VOLUME_ELEVATED, start_at, end_at)
                elif sample_rms_delta <= VOLUME_LOWERED_VARIANCE_THRESHOLD:
                    self.record_state_change(STATE_VOLUME_LOWERED, start_at, end_at)
                elif self.current_state != STATE_VOLUME_CONSTANT:
                    self.record_state_change(STATE_VOLUME_CONSTANT, start_at, end_at)
                    
                logging.debug("frame {},{},{},{},{},{},{},{},{},{},{},{},{}".format(seq, chunk_size, round(start_at, 2), round(end_at, 2), round((end_at - start_at), 4), len(sound_bite), get_max(sound_bite), window_rms, sample_rms, len(self._sound_samples), volume_silence_threshold, sample_rms_delta, sample_rms_variance))
                #self.plot_recent_samples_rms()
            except EOFError:
                break
            except Exception:
                logging.exception("error consuming audio")
                break
        logging.info("Done consuming audio")
