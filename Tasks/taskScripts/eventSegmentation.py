# Edited movieTask script for Event Segmentation (NO ESQ)
from psychopy import visual 
import psychopy
psychopy.prefs.hardware['audioLib'] = ['sounddevice', 'pyo','pygame']
from matplotlib.pyplot import pause

import pandas as pd
from psychopy import gui, data, core, event
import os.path
import csv
import random
from datetime import datetime
from psychopy import prefs

from psychopy.constants import NOT_STARTED, STARTED, FINISHED

###################################################################################################

def save_comp_csv(responses_data, participant_id, clipname, seed):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(current_directory, "..", "comp_file")
    os.makedirs(log_folder, exist_ok=True)

    current_datetime = datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")        
    csv_path = os.path.join(log_folder, f"{participant_id}_{clipname}_{seed}_{current_datetime}_comp_output.csv")
    
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['idno', 'videoname', 'qnumber', 'response', 'correctness']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(responses_data)

def present_comprehension_question(win, stim, question_number, participant_id, videoname, responses_data):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    questions_file_path = os.path.join(current_directory, "resources", "Movie_Task", "csv", "comprehension_questions.csv")
    
    # Load questions from CSV
    with open(questions_file_path, 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        questions = list(csv_reader)

    question_data = questions[question_number - 1]
    question_text = question_data['question']
    options = question_data['options'].split('|')
    correct_option = int(question_data['correct'])

    # Format display
    question_text = f"{question_text}\n"
    for idx, option in enumerate(options, start=1):
        question_text += f"{option}\n"

    stim.setText(question_text)
    stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=[str(i) for i in range(1, len(options) + 1)])
    response = keys[0]

    correctness = "correct" if int(response) == correct_option else "incorrect"

    responses_data.append({
        'idno': participant_id,
        'videoname': videoname,
        'qnumber': question_number,
        'response': response,
        'correctness': correctness
    })
    return responses_data

def present_seen_question(win, stim, question_number, participant_id, videoname, responses_data):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    questions_file_path = os.path.join(current_directory, "resources", "Movie_Task", "csv", "comprehension_questions.csv")

    # Load questions from CSV
    with open(questions_file_path, 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        questions = list(csv_reader)

    if question_number < 1 or question_number > len(questions):
        return responses_data  

    question_data = questions[question_number - 1]
    question_text = question_data['question']
    options = question_data['options'].split('|')

    # Format display
    question_text = f"{question_text}\n"
    for idx, option in enumerate(options, start=1):
        question_text += f"{option}\n"

    stim.setText(question_text)
    stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=[str(i) for i in range(1, len(options) + 1)])
    
    if not keys:
        return responses_data

    response = keys[0]

    responses_data.append({
        'idno': participant_id,
        'videoname': videoname,
        'qnumber': question_number,
        'response': response
    })
    return responses_data

def save_seen_csv(responses_data, participant_id, clipname, seed):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(current_directory, "..", "seen_file")
    os.makedirs(log_folder, exist_ok=True)

    current_datetime = datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")        
    csv_path = os.path.join(log_folder, f"{participant_id}_{clipname}_{seed}_{current_datetime}_seen_output.csv")
    
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['idno', 'videoname', 'qnumber', 'response']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        if responses_data:
            csv_writer.writerows(responses_data)

###################################################################################################

def runexp(filename, timer, win, writer, resdict, runtime, dfile, seed, probever, participant_id):
    writera = writer[1]
    writer = writer[0]
    random.seed(seed)
    
    resdict['Timepoint'], resdict['Time'] = 'Movie Task Start', timer.getTime()
    writer.writerow(resdict)
    resdict['Timepoint'], resdict['Time'] = None, None

    responses_data = []
    seen_data = []

    instructions = """For the next phase of the experiment, you will watch a series of video clips.

While you watch the videos, please press the SPACE bar whenever you identify an EVENT BOUNDARY (the same way you have practiced earlier).

In the videos you are about to watch, it is important to note that every scene change might not be an event boundary. However, sometimes and event boundary might occur during a scene (e.g. if one theme ends and another begins).

Remember, an event boundary occurs where you perceive one event finishes and another begins.
"""
    stim = visual.TextStim(win, "", color=[-1,-1,-1], wrapWidth=1300, units="pix", height=40)
    stim.setText(instructions)
    stim.draw()
    win.flip()
    event.waitKeys(keyList=['return'])

    # Load video and probe order
    trialvideo = os.path.join(os.getcwd(), "taskScripts", filename[1])
    trialsplits = pd.read_csv(os.path.join(os.getcwd(), "taskScripts", filename[0]))
    videoname = filename[1].rsplit("/",1)[-1]
    trialname = "Movie Task-" + trialvideo.split(".")[0].split("/")[-1]

    # Init movie
    resdict['Timepoint'], resdict['Time'], resdict['Auxillary Data'] = 'Movie Start', timer.getTime(), videoname
    writer.writerow(resdict)
    resdict['Timepoint'], resdict['Time'], resdict['Auxillary Data'] = None, None, None
    
    mov = visual.MovieStim3(win, trialvideo, size=(1920, 1080), loop=False)
    movieClock = core.Clock()

    while mov.status != STARTED:
        mov.draw()
        win.flip()
    
    movieClock.reset()
    

    # Create boundary CSV for this run
    event_seg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "event_seg", 
        f"{participant_id}_{filename[1].split('/')[-1].split('.')[0]}_{seed}_{datetime.now().strftime('%Y_%m_%d-%p%I_%M_%S')}_boundaries.csv")
    os.makedirs(os.path.dirname(event_seg_path), exist_ok=True)
    with open(event_seg_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["ParticipantID", "VideoName", "BoundaryTime(s)"])
    
    # Movie loop
    while mov.status != visual.FINISHED:
        mov.draw()
        win.flip()
        keys = event.getKeys(timeStamped=movieClock)
        for key, timestamp in keys:
            print(f"[DEBUG] Key: {key}, Timestamp: {timestamp}, Movie status: {mov.status}, Clock time: {movieClock.getTime()}") #debug
            if key == 'space':
                with open(event_seg_path, "a", newline="") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([participant_id, videoname, round(timestamp, 3)])

     #at the end of each clip, present comprehension questions
    video_basename = os.path.basename(filename[1])  # e.g., "c4.mp4"
    if video_basename == "prestige.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 1, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 2, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 3, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 4, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 33, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
        #core.quit() #use this to debug

    if video_basename == "pulpFiction.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 5, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 6, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 7, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 8, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 34, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    if video_basename == "shawshank.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 9, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 10, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 11, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 12, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 35, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    if video_basename == "backToFuture.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 13, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 14, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 15, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 16, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 36, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)

    if video_basename == "12_years.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 17, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 18, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 19, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 20, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 37, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    if video_basename == "lms.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 21, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 22, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 23, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 24, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 38, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)

    if video_basename == "c4.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 25, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 26, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 27, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 28, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 39, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    if video_basename == "500Days.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 29, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 30, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 31, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 32, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 40, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    

    return trialname

def run_practice(win):
    
    stim = visual.TextStim(win, "", color=[-1, -1, -1], wrapWidth=1300, units="pix", height=40)
    practice_video_path = os.path.join(os.getcwd(), "taskScripts", "resources", "Movie_Task", "videos", "practice_clip.mp4")
    min_required_presses = 3
    max_attempts = 2
    attempt_count = 0
    passed = False

    start_screen = """In the following experiment, you will watch a series of video clips. As you view the clips, please pay attention to the flow of events, specifically, please watch for natural breaks or transitions in the videos where you feel one event ends and another starts. 
    
                    \nThis 'gap' is commonly referred to as EVENT BOUNDARY. An event boundary occurs where you perceive one event finishes and another begins. We ask you to identify the event boundaries by pressing SPACE on your keyboard.

                    \nThere are no right or wrong answers in this task; we are interested in your personal perception of when and where events transition.

                    \nWe will first begin with a practice trial.
                    """

    stim.setText(start_screen)
    stim.draw()
    win.flip()
    
    # Wait for user to press enter to continue. 
    event.waitKeys(keyList=(['return']))


    while attempt_count < max_attempts and not passed:
        stim.setText("""In the following practice phase, you will watch a short video, where there are several clear EVENT BOUNDARIES.

Please press SPACE when you observe an event boundary.

Remember, an event boundary occurs where you perceive one event finishes and another begins.""")
        stim.draw()
        win.flip()
        event.waitKeys(keyList=['return'])

        practice_mov = visual.MovieStim3(win, practice_video_path, size=(1920, 1080), flipVert=False, flipHoriz=False, loop=False)
        clock = core.Clock()
        boundaries = []
        

        while practice_mov.status != visual.FINISHED:
            practice_mov.draw()
            win.flip()
            keys = event.getKeys(timeStamped=clock)
            for key, t in keys:
                if key == 'space':
                    boundaries.append(round(t, 3))

        if len(boundaries) >= min_required_presses:
            passed = True
        else:
            attempt_count += 1
            if attempt_count < max_attempts:
                retry = """It appears that you failed to identify some event boundaries in the first practice phase.

Typically, participants identify about 3 or 4 event boundaries in this practice video.

You now have the chance to try again."""
                stim.setText(retry)
                stim.draw()
                win.flip()
                event.waitKeys(keyList=['return'])
            else:
                retry = """You failed to identify adequate event boundaries in the practice phase.

Please alert the experimenter.

(Do NOT close this window)"""
                stim.setText(retry)
                stim.draw()
                win.flip()
                while True:
                    keys = event.getKeys()
                    if 'r' in keys:
                        attempt_count = 0  # reset the attempt count
                        passed = False     # reset passed flag
                        break              # restart the practice loop
                    elif 'q' in keys:
                        core.quit()
                    core.wait(1)
