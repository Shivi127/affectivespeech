import re
import sys
import multiprocessing
import logging
_DEBUG = logging.INFO
import multiprocessingloghandler

import traceback
import threading
from queue import Queue
from queue import Empty

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

import audioop
import pyaudio
import time
from six.moves import queue
from datetime import datetime
from collections import deque

import sound_processor

from show_text_fe import show_text
from sound_state import *

# Audio recording parameters
RATE = 16000
CHUNK_DURATION_SECS = 0.10  # 100 ms chunks
CHUNK = int(RATE * CHUNK_DURATION_SECS)

PLOT_HISTORY_SECS = 2
_PLOT_HISTORY_COUNT = int(PLOT_HISTORY_SECS / CHUNK_DURATION_SECS)

CAPTION_DURATION_SECS = 60.0

STATE_SAMPLE_LIFETIME_SECS = 5
PAUSE_MINIMUM_SPAN_SECS = 2


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk, sound_chunk_pipe=None):
        self._rate = rate
        self._chunk = chunk
        self._sound_chunk_pipe = sound_chunk_pipe

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

        self._audio_base_time = None

    def get_start_time(self):
        return self._audio_base_time

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def _set_base_time(self):
        if self._audio_base_time is None:
            self._audio_base_time = time.time() - CHUNK_DURATION_SECS

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._set_base_time()
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        frame_nbr = 0
        start_time = None
        end_time = None
        chunk_count = 0
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            received_at = time.time()
            if end_time and received_at < end_time:
                logging.error('derived chunk start time < previous chunk end time')
            chunk_count += 1
            # Derive the start time from the first chunk of a buffered set 
            if not start_time:
                start_time = received_at - CHUNK_DURATION_SECS

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    received_at = time.time()
                    if chunk is None:
                        logging.warning('null audio chunk')
                        return
                    data.append(chunk)
                    chunk_count += 1
                except queue.Empty:
                    break

            frame_nbr += 1
            sound_chunk = b''.join(data)
            end_time = received_at
            soundbite = (sound_chunk, frame_nbr, chunk_count, start_time, end_time)
            start_time = None
            chunk_count = 0
            if self._sound_chunk_pipe:
                self._sound_chunk_pipe.send(soundbite)
            logging.debug('sent')
            yield soundbite

def parse_timestamp_to_secs(timestamp):
    span = timestamp.seconds
    nanos = timestamp.nanos
    span += float(nanos) / 1000000000.0
    return span
 

def listen_print_loop(responses, caption_file, sound_consumer, audio_start_time):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    If a caption file is specified, print final responses to a caption file
    with timestamps every minute.
    """
    last_phrase = ''
    last_caption_timestamp = 0
    for response in responses:
        logging.info("response: {}".format(response))
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue
        logging.info("result: {}".format(result)) 
        result_end_time = audio_start_time + parse_timestamp_to_secs(result.result_end_time)

        # Display the transcription of the top alternative.
        words = result.alternatives[0].words

        times = []
        for word in words:
            start =  audio_start_time + parse_timestamp_to_secs(word.start_time)
            end =  audio_start_time + parse_timestamp_to_secs(word.end_time)
            duration = (start, end, (end-start))
            times.append((word.word, duration))
        # Handle words only being returned for final results.  https://issuetracker.google.com/issues/144757737
        if words:
            logging.info('word times: %s', [(word[0], time.strftime('%H:%M:%S', time.localtime(word[1][1]))) for word in times])
            logging.info("word %s at %s", phrase.split(" ")[-1:], time.strftime('%H:%M:%S', time.localtime(result_end_time)))
            phrase = " ".join([word.word for word in words])
        else:
            phrase = result.alternatives[0].transcript
            logging.info("word %s at %s", phrase.split(" ")[-1:], time.strftime('%T', time.localtime(result_end_time)))
        if caption_file and result.is_final:
            caption = phrase+'\n'
            if time.time() - last_caption_timestamp > CAPTION_DURATION_SECS:
                caption = "{} {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), caption)
                last_caption_timestamp = time.time()
            caption_file.write(caption)

        logging.info('state: %s', str(sound_consumer.current_state))
        show_text(phrase, sound_consumer.current_state)
        last_phrase = phrase

        # Exit recognition if our exit word is said 3 times
        if result.is_final and len(re.findall(r'quit', phrase, re.I)) == 3:
            print('Exiting..')
            show_text('Exiting..')
            break


def main(argv):
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en-US'  # a BCP-47 language tag

    if len(argv) > 1:
        caption_file = open(argv[1],"w")
    else:
        caption_file = None
 
    client = speech.SpeechClient()

    log_stream = sys.stderr
    log_queue = multiprocessing.Queue(100)
    handler = multiprocessingloghandler.ParentMultiProcessingLogHandler(logging.StreamHandler(log_stream), log_queue)
    logging.getLogger('').addHandler(handler)
    logging.getLogger('').setLevel(_DEBUG)

    ipc_pipe =  multiprocessing.Pipe()
    sound_send_pipe, unused_sound_pipe = ipc_pipe
    sound_consumer = sound_processor.SoundRenderer(ipc_pipe, log_queue, logging.getLogger('').getEffectiveLevel(), _PLOT_HISTORY_COUNT)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_word_time_offsets=True)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    sound_consumer.start()
    with MicrophoneStream(RATE, CHUNK, sound_send_pipe) as stream:
        while stream.get_start_time() is None:
          time.sleep(CHUNK_DURATION_SECS)
        audio_generator = stream.generator()
        while True:
          requests = (types.StreamingRecognizeRequest(audio_content=content)
             for content, seq, chunk_count, start_offset, end_offset in audio_generator)
          responses = client.streaming_recognize(streaming_config, requests)
          try:
            listen_print_loop(responses, caption_file, sound_consumer, stream.get_start_time())
          except:
            traceback.print_exc()
          finally:
            break
    if caption_file:
      caption_file.close()
    unused_sound_pipe.close()
    sound_send_pipe.close()
    logging.info("stopping background process")
    sound_consumer.stop()
    sound_consumer.join()
    logging.info("logged: main done")
    logging.shutdown()

    print("ended")
    sys.exit(0)

if __name__ == '__main__':
    main(sys.argv)
