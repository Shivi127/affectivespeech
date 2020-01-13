import sys
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

PLOT_HISTORY_SECS = 2
_PLOT_HISTORY_COUNT = int(PLOT_HISTORY_SECS / CHUNK_DURATION_SECS)

STATE_SAMPLE_LIFETIME_SECS = 5
PAUSE_MINIMUM_SPAN_SECS = 2

def get_max(audio_chunk):
    return audioop.max(audio_chunk, 2)

def get_rms(audio_chunk):
    return audioop.rms(audio_chunk, 2)

def get_avg(audio_chunk):
    return audioop.avg(audio_chunk, 2)

class SoundConsumer(Background):
    def add_to_recent_samples(self, sample):
        self._sound_samples.append(sample)

    def calculate_function_average_above_threshold_for_samples(self, function, threshold):
        applied_function_results = [eligible_sample for eligible_sample in [function(sample) for sample in self._sound_samples if sample is not None] if eligible_sample > threshold]
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
        sys.stderr.write('min of {}\n'.format(applied_function_results))
        return min(applied_function_results)

    def __init__(self, audio_chunk_queue):
        self.__stop_flag = False
        self.current_state = STATE_VOLUME_CONSTANT
        self._audio_chunk_queue = audio_chunk_queue
        self._sound_samples = deque()
        self._state_changes = deque()
        self.current_pause_start = None
        self.current_pause_end = None

    def stop(self):
        self.__stop_flag = True

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

    def plot_recent_samples_rms(self, sample_count):
        slice_start = -1 * min(sample_count, len(self._sound_samples))
        plot_samples = list(self._sound_samples)[slice_start:]
        plots = [get_rms(sample) for sample in plot_samples if sample is not None]
        sys.stderr.write('plot: {}\n'.format(plots))
        draw_bar('sound level', 'RMS', plots)

    def consume_raw_audio(self):
        sys.stderr.write("Waiting to consume audio\n")
        sys.stderr.flush()
        while not self.__stop_flag:
            try:
                seq, chunk_size, start_at, end_at, sound_bite = self._audio_chunk_queue.get(block=False)
                if sound_bite is None:
                    sys.stderr.write('NULL audio chunk\n')
                sample_rms = get_rms(sound_bite)
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
                        sys.stderr.write('discard {} samples\n'.format(len(self._sound_samples)))
                        self._truncate_recent_samples()
 
                window_rms = self.calculate_function_average_above_threshold_for_samples(get_rms, volume_silence_threshold)
                if window_rms is None:
                    window_rms = sample_rms
                sample_rms_delta = sample_rms - window_rms

                sample_rms_variance = float(sample_rms_delta) / window_rms

                if sample_rms_variance >= VOLUME_RAISED_VARIANCE_THRESHOLD:
                    self.record_state_change(STATE_VOLUME_ELEVATED, start_at, end_at)
                elif sample_rms_delta <= VOLUME_LOWERED_VARIANCE_THRESHOLD:
                    self.record_state_change(STATE_VOLUME_LOWERED, start_at, end_at)
                elif self.current_state != STATE_VOLUME_CONSTANT:
                    self.record_state_change(STATE_VOLUME_CONSTANT, start_at, end_at)
                    
                sys.stderr.write("frame {},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(seq, chunk_size, round(start_at, 2), round(end_at, 2), round((end_at - start_at), 4), len(sound_bite), get_max(sound_bite), window_rms, sample_rms, len(self._sound_samples), volume_silence_threshold, sample_rms_delta, sample_rms_variance))
                self.plot_recent_samples_rms(_PLOT_HISTORY_COUNT)
            except Empty:
                pass
        sys.stderr.write("Done consuming audio\n")
        sys.stderr.flush()
