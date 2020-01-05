#!/bin/bash
cmd_dir=`dirname $0`
PYTHON=$(which python3)

cd $cmd_dir
export GOOGLE_APPLICATION_CREDENTIALS=`ls ../piday*json | head -1`
echo $GOOGLE_APPLICATION_CREDENTIALS
$PYTHON show_text_fe.py 'waiting for IP'
while :
do
  sleep 1
  ping -c 1 www.github.com 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    break
  fi
done
MY_ADDRESS=$(ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}')
$PYTHON show_text_fe.py "My address is $MY_ADDRESS"
sleep 2
$PYTHON show_text_fe.py "quit quit quit to exit"

TIMESTAMP=$(date  +"%m-%d-%H-%M")
transcript="${HOME}/Desktop/SPEECH-${TIMESTAMP}.txt"
$PYTHON show_text_fe.py "caption file is: ${transcript}"
$PYTHON ./streaming_recognizer.py "${transcript}" 2>/tmp/speech.err
rc=$?
if [[ $rc -eq 0 ]]; then
  $PYTHON show_text_fe.py 'shutting down'
fi
