Voice transcriber
For Pi Day 2019
2019, Raymond Blum (raygeeknyc@)

This is a handy Raspberry Pi device that uses the Google Cloud Speech service
to provide live transcription of detected speech.

Output is scrolled, Karaoke style, on an attached Nokia LCD display or to the console.
- if autologin is set up this will be shown on an attached monitor.

Setup quickstart:
1) These files should reside in ~pi/workspace/piday
2) A client key for a Google Cloud account for which the Google Speech API is enabled should be placed in ~pi/workspace with a name matching '*piday*/.json'.
3) The directory /boot/files/ should contain the file networks_list.txt which has a list of WiFi networks for the device to connect to.
  you have to create that directory first.
  The file is processed when the device boots.

The boot filesystem should be mounted by most PCs when the SD card is inserted
