from show_text_fe import show_text

import re
import sys

import traceback
import threading
from Queue import Queue
from Queue import Empty

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import audioop
import pyaudio
from six.moves import queue
import time
from datetime import datetime

# Audio recording parameters
RATE = 16000
CHUNK_DURATION_SECS = 0.10  # 100 ms chunks
CHUNK = int(RATE * CHUNK_DURATION_SECS)

TIMESTAMP_PERIOD_SECS = 60.0

class SoundConsumer(object):
   def __init__(self, audio_chunk_queue):
      self.__stop_flag = False
      self._audio_chunk_queue = audio_chunk_queue

   def stop(self):
      self.__stop_flag = True

   def consume_raw_audio(self):
      sys.stderr.write("Waiting to consume audio\n")
      sys.stderr.flush()
      while not self.__stop_flag:
         try:
            seq, chunk_count, start_at_elapsed, end_at_elapsed, sound_bite = self._audio_chunk_queue.get(block=False)
            sys.stderr.write("got frame {}, chunk count {}, start {}, end {}, duration {}, {} audio bytes max volume {}\n".format(seq, chunk_count, round(start_at_elapsed, 2), round(end_at_elapsed, 2), round((end_at_elapsed - start_at_elapsed), 4), len(sound_bite), audioop.max(sound_bite, 2)))
         except Empty:
            pass
      sys.stderr.write("Done consuming audio\n")
      sys.stderr.flush()

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk, chunk_queue=None):
        self._rate = rate
        self._chunk = chunk
        self._chunk_queue = chunk_queue

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

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
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        frame_nbr = 0
        first_audio_start_time = time.time()
        start_elapsed_time = None
        chunk_count = 0
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            chunk_count += 1
            if not start_elapsed_time:
                start_elapsed_time = time.time() - CHUNK_DURATION_SECS - first_audio_start_time

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        sys.stderr.write('null audio chunk\n')
                        sys.stderr.flush()
                        return
                    data.append(chunk)
                    chunk_count += 1
                except queue.Empty:
                    break

            frame_nbr += 1
            sound_chunk = b''.join(data)
            end_elapsed_time = time.time() - first_audio_start_time
            soundbite = (frame_nbr, chunk_count, start_elapsed_time, end_elapsed_time, sound_chunk)
            start_elapsed_time = None
            chunk_count = 0
            if self._chunk_queue:
                self._chunk_queue.put(soundbite)
            yield soundbite

def parse_time(timestamp):
    span = timestamp.seconds
    nanos = timestamp.nanos
    span += float(nanos) / 1000000000.0
    return span
 

def listen_print_loop(responses, caption_file):
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
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue
      
        # Display the transcription of the top alternative.
        words = result.alternatives[0].words

        times = []
        for word in words:
            start = parse_time(word.start_time)
            end = parse_time(word.end_time)
            span = (start, end, (end-start))
            times.append((word.word, span))
        # Handle words only being returned for final results.  https://issuetracker.google.com/issues/144757737
        if words:
            sys.stderr.write('word times: {}\n'.format(times))
            phrase = " ".join([word.word for word in words])
        else:
            phrase = result.alternatives[0].transcript
        if caption_file and result.is_final:
            caption = phrase+'\n'
            if time.time() - last_caption_timestamp > TIMESTAMP_PERIOD_SECS:
                caption = "{} {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), caption)
                last_caption_timestamp = time.time()
            caption_file.write(caption)

        show_text(phrase)
        last_phrase = phrase

        # Exit recognition if our exit word is said 3 times
        if len(re.findall(r'quit', phrase, re.I)) == 3:
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

    sound_content = Queue()
    sound_processor = SoundConsumer(sound_content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_word_time_offsets=True)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    sound_processor_worker = threading.Thread(target=sound_processor.consume_raw_audio)
    sound_processor_worker.start()
    with MicrophoneStream(RATE, CHUNK, sound_content) as stream:
        audio_generator = stream.generator()
        while True:
          requests = (types.StreamingRecognizeRequest(audio_content=content)
             for seq, chunk_count, start_offset, end_offset, content in audio_generator)
          responses = client.streaming_recognize(streaming_config, requests)
          try:
            listen_print_loop(responses, caption_file)
            break
          except:
            traceback.print_exc()
          finally:
            if caption_file:
              caption_file.close()
            sound_processor.stop()
    print "ended"
    quit()

if __name__ == '__main__':
    main(sys.argv)
