import sys
import logging

from queue import Queue
from queue import Empty

from background_process import Background

import audioop
import time
from datetime import datetime
from collections import deque

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

class SoundRenderer(Background):
    def run(self):
        self._initLogging()
        logging.debug('run')
        self.consume_audio_and_text()

    def add_to_recent_window_rms(self, window_rms):
        self._sound_windows.append(window_rms)

    def add_to_recent_samples(self, sample, timestamp):
        self._sound_samples.append(sample)
        self._sample_timestamps.append(timestamp)

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

    def __init__(self, audio_chunk_pipe, log_queue, log_level, plot_sample_count):
        super(SoundRenderer, self).__init__(audio_chunk_pipe, log_queue, log_level)
        self.current_state = STATE_VOLUME_CONSTANT
        self._sound_samples = deque(maxlen=plot_sample_count)
        self._sample_timestamps = deque(maxlen=plot_sample_count)
        self._sound_windows = deque(maxlen=plot_sample_count)
        self._state_changes = deque(maxlen=plot_sample_count)
        self.plot_sample_count = plot_sample_count
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

    def plot_recent_samples_rms(self):
        slice_start = -1 * min(self.plot_sample_count, len(self._sound_samples))
        samples_plot = [get_rms(sample) for sample in list(self._sound_samples)[slice_start:]]
        windows_plot = [window for window in list(self._sound_windows)[slice_start:]]
        logging.debug('plot: {} {}'.format(samples_plot, windows_plot))
        sample_labels = []
        prev_timestamp = None
        for timestamp in list(self._sample_timestamps)[slice_start:]:
            if prev_timestamp is None:
                prev_timestamp = timestamp
            if int(timestamp) != int(prev_timestamp):
                sample_labels.append(time.strftime('%T', time.gmtime(timestamp)))
                prev_timestamp = timestamp
            else:
                sample_labels.append('')
        draw_graph('sound level', (self.all_min, self.all_max), 'RMS', samples_plot, windows_plot, sample_labels)

    def consume_audio_and_text(self):
        logging.info('Waiting to consume')
        while not self._exit.is_set():
            try:
                logging.debug('recv')
                packet = self._receive_pipe.recv()
                if isinstance(packet[0], str):
                    self.process_text(packet)
                else:
                    self.process_audio(packet)
            except EOFError:
                break
            except Exception:
                logging.exception('error consuming audio')
                break
        logging.info('Done consuming audio')

    def process_text(self, packet):
        pass

    def process_audio(self, packet):
        seq, chunk_size, start_at, end_at, sound_bite = packet
        if sound_bite is None:
            logging.debug('NULL audio chunk')
        sample_rms = get_rms(sound_bite)
        self.all_min = min(sample_rms, self.all_min)
        self.all_max = max(sample_rms, self.all_max)
        window_min = self.calculate_function_min_for_samples(get_rms)
        if window_min is None:
            window_min = sample_rms
        volume_silence_threshold = window_min * VOLUME_SILENCE_RANGE
        self.add_to_recent_samples(sound_bite, start_at)

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
            
        logging.debug('frame {},{},{},{},{},{},{},{},{},{},{},{},{}'.format(seq, chunk_size, round(start_at, 2), round(end_at, 2), round((end_at - start_at), 4), len(sound_bite), get_max(sound_bite), window_rms, sample_rms, len(self._sound_samples), volume_silence_threshold, sample_rms_delta, sample_rms_variance))
        self.plot_recent_samples_rms()
