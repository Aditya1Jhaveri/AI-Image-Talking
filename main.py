from config import *
from speech import generate_speech
from image import generate_image
from lips import modify_lips
import humanize
import datetime as dt
from argparse import ArgumentParser
import shutil

import os
import glob
from improve import improve, vid2frames, restore_frames
from animate_face import animate_face

# message = """Over the holiday season, capturing photos and videos of the festivities with family and friends 
# is an important activity for many. The iPhone has a suite of camera features that can significantly elevate 
# the quality and creativity of your holiday photos and videos."""
# message = """Apple today confirmed that it will be permanently closing its Infinite Loop retail store in 
# Cupertino, California on January 20. Infinite Loop served as Apple's headquarters between the mid-1990s and 
# 2017, when its current Apple Park headquarters opened a few miles away."""


def main():
		# Parse the arguments
	parser = ArgumentParser()
	parser.add_argument("--improve", action="store_true", help="use Real ESRGAN to improve the video")
	parser.add_argument("--skipgen", action="store_true", help="improve the video only")
	parser.add_argument("--path_id", default=str(int(time.time())), help="set the path id to use")
	parser.add_argument("--message_file", type=str,  help="path to the file containing the speech message")
	parser.add_argument("--voice", type=str, help="path to speaker voice file")
	parser.add_argument("--lang",  type=str, help="select the language for speaker voice")
	parser.add_argument("--speech",  help="path to WAV speech file")
	parser.add_argument("--image_prompt",  help="path to the file containing the image message")
	parser.add_argument("--image", default=imgfile, help="path to avatar file")

	args = parser.parse_args()
	tstart = time.time()
    
	## SET PATH
	path_id = args.path_id
	path = os.path.join("temp", path_id)
	print("path_id:", path_id, "path:", path)
	os.makedirs(path, exist_ok=True)
	outfile = os.path.join("results", path_id + "_small.mp4")
	finalfile = os.path.join("results", path_id + "_large.mp4")
 
	# Initialize variables
	message = None
	prompt = None

	# Conditionally read the message file
	if args.message_file:
		message = read_message_from_file(args.message_file)

	# Conditionally read the prompt file
	if args.image_prompt:
		prompt = read_prompt_from_file(args.image_prompt)
  
	input_voice = args.voice
	input_lang = args.lang

# Read the message from the specified file
	if not args.skipgen:
		if args.message_file and args.voice and args.lang:
      
		## GENERATE SPEECH
			tspeech = "None"
			print("-----------------------------------------")
			print("generating speech")
			t0 = time.time()
			generate_speech(path_id, tts_output, message,input_voice, input_lang)
			tspeech = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t0)))
			print("\ngenerating speech:", tts_output,"in",tspeech)
		else:
			print("using:", args.speech)
			shutil.copyfile(args.speech, os.path.join("temp", path_id, audiofile))

##Generate the Avatar image if it's not provided

		## GENERATE AVATAR IMAGE
		timage = "None"
		if args.image == imgfile:

			print("-----------------------------------------")
			print("generating avatar image")
			t1 = time.time()
			avatar_description = prompt
			generate_image(path_id, imgfile, f"hyperrealistic digital avatar, centered, {avatar_description}, \
						rim lighting, studio lighting, looking at the camera")
			timage = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t1)))
			print("\ngenerating avatar:", timage)
		else:
			shutil.copyfile(args.image, os.path.join("temp", path_id, imgfile))

		## ANIMATE AVATAR IMAGE

		print("-----------------------------------------")
		print("animating face with driver")
		t2 = time.time()	
		# audiofile determines the length of the driver movie to trim
		# driver movie is imposed on the image file to produce the animated file
		animate_face(path_id, tts_output, driverfile, imgfile, animatedfile)
		tanimate = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t2)))
		print("\nanimating face:", tanimate)

		## MODIFY LIPS TO FIT THE SPEECH

		print("-----------------------------------------")
		print("modifying lips")
		t3 = time.time()
		os.makedirs("results", exist_ok=True)
		
		modify_lips(path_id, tts_output, animatedfile, outfile)
		tlips = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t3)))
		print("\nmodifying lips:", tlips)

	## IMPROVE THE OUTPUT VIDEO
	if args.improve:
		t4 = time.time()
		print("-----------------------------------------")
		print("converting video to frames")
		shutil.rmtree(os.path.join(path, "improve"), ignore_errors=True)
		os.makedirs(os.path.join(path, "improve", "disassembled"), exist_ok=True)
		os.makedirs(os.path.join(path, "improve", "improved"), exist_ok=True)	
		
		vid2frames(outfile, os.path.join(path, "improve", "disassembled"))
		print("-----------------------------------------")
		print("improving face")
		improve(os.path.join(path, "improve", "disassembled"), os.path.join(path, "improve", "improved"))
		print("-----------------------------------------")
		print("restoring frames")
		
		restore_frames(os.path.join(path, tts_output), finalfile, os.path.join(path, "improve", "improved"))		
		timprove = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t4)))
		print("\nimproving video:", timprove)
	
	print("done")
	print("Overall timing")
	print("--------------")
	if not args.skipgen:
		print("generating speech:", tspeech)
		print("generating avatar image:", timage)
		print("animating face:", tanimate)
		print("modifying lips:", tlips)
	if args.improve:
		print("improving finished video:", timprove)
	print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - tstart))))

if __name__ == '__main__':
	main()