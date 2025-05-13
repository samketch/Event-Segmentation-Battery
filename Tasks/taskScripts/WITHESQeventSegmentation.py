#Written by Brontë McKeown and Theodoros Karapanagiotidis
#This is an edited version of the movieTask script that runs an event segmentation task.
from psychopy import visual 
import psychopy
psychopy.prefs.hardware['audioLib'] = ['sounddevice', 'pyo','pygame']
from matplotlib.pyplot import pause

import pandas as pd
from psychopy import gui, data, core,event
from taskScripts import ESQ
import os.path
import csv

import random

from datetime import datetime
from psychopy import prefs



###################################################################################################

def save_comp_csv(responses_data, participant_id, clipname, seed):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(current_directory, "..", "comp_file")

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
     
    with open(questions_file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if "�" in line:
                print(f"Line {lineno}: {line.strip()}")

    # Load questions from the CSV file
    with open(questions_file_path, 'r', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        questions = list(csv_reader)

    question_data = questions[question_number - 1]
    question_text = question_data['question']
    options = question_data['options'].split('|')
    correct_option = int(question_data['correct'])  # Convert to an integer

    # Present the question
    question_text = f"{question_text}\n"
    for idx, option in enumerate(options, start=1):
        question_text += f"{option}\n"

    stim.setText(question_text)
    stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=[str(i) for i in range(1, len(options) + 1)])
    response = keys[0]  # Store the selected option

    correctness = "correct" if int(response) == correct_option else "incorrect"

    # Store the response data
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
    
    with open(questions_file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if "�" in line:
                print(f"Line {lineno}: {line.strip()}")

    # Load questions from the CSV file
    with open(questions_file_path, 'r', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        questions = list(csv_reader)
    
    #validate question_number
    total_questions = len(questions)
    if question_number < 1 or question_number > total_questions:
        return responses_data  # Return without modifying responses_data

    question_data = questions[question_number - 1]
    question_text = question_data['question']
    options = question_data['options'].split('|')

    # Present the question
    question_text = f"{question_text}\n"
    for idx, option in enumerate(options, start=1):
        question_text += f"{option}\n"

    stim.setText(question_text)
    stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=[str(i) for i in range(1, len(options) + 1)])
    
    if not keys:
            print('no response received')
            return responses_data

    response = keys[0]  # Store the selected option

    # Store the response data
    responses_data.append({
        'idno': participant_id,
        'videoname': videoname,
        'qnumber': question_number,
        'response': response
    })

    # Debugging: Print the updated responses_data list
    print(f"Current seen responses: {responses_data}")
    
    return responses_data

def save_seen_csv(responses_data, participant_id, clipname, seed):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(current_directory, "..", "seen_file")

    os.makedirs(log_folder, exist_ok=True) #create file if it does not exist

    current_datetime = datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")
        
    csv_path = os.path.join(log_folder, f"{participant_id}_{clipname}_{seed}_{current_datetime}_seen_output.csv")
    
    print('saving seen responses', responses_data) #debug: print response data before saving

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['idno', 'videoname', 'qnumber', 'response']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        if responses_data: #only write if there are responses
            csv_writer.writerows(responses_data)
        else:
            print('Warning: No responses to save')

def save_eventseg_csv(boundary_times, participant_id, clipname, seed):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(current_directory, "..", "event_seg")
    os.makedirs(log_folder, exist_ok=True)

    current_datetime = datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")
    filename = f"{participant_id}_{clipname}_{seed}_{current_datetime}_boundaries.csv"
    csv_path = os.path.join(log_folder, filename)

    print(f"[EVENT SEG] Saving to: {csv_path}")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ParticipantID", "VideoName", "BoundaryTime(s)"])
        for t in boundary_times:
            writer.writerow([participant_id, clipname, t])


def runexp(filename, timer, win, writer, resdict, runtime,dfile,seed,probever, participant_id):
    writera = writer[1]
    writer = writer[0]
    random.seed(seed)
    
    resdict['Timepoint'], resdict['Time'] = 'Movie Task Start', timer.getTime()
    writer.writerow(resdict)
    resdict['Timepoint'], resdict['Time'] = None,None

    responses_data = []
    seen_data = []
    
    

    

    # user can update instructions for task here if required.
    instructions =      """For the next phase of the experiment, you will watch a series of video clips.

While you watch the videos, please press the SPACE bar whenever you identify an EVENT BOUNDARY (the same way you have practiced earlier).

In the videos you are about to watch, it is important to note that every scene change might not be an event boundary. However, sometimes and event boundary might occur during a scene (e.g. if one theme ends and another begins).

Remember, an event boundary occurs where you perceive one event finishes and another begins.

                        """
    

    # create text stimuli to be updated for start screen instructions.
    stim = visual.TextStim(win, "", color = [-1,-1,-1], wrapWidth = 1300, units = "pix", height=40)

    # update text stim to include instructions for task. 
    stim.setText(instructions)
    stim.draw()
    win.flip()
    # Wait for user to press enter to continue. 
    event.waitKeys(keyList=(['return']))

    # update text stim to include start screen for task. 
    #stim.setText(start_screen)
    #stim.draw()
    #win.flip()
    
    # Wait for user to press enter to continue. 
    event.waitKeys(keyList=(['return']))
    
    
    # Create two lists, one with the control videos, and one with action videos
    # Videos are sorted based on their file name
    #list_of_videos = os.listdir(os.path.join(os.getcwd(), 'taskScripts//resources//Movie_Task//videos'))
    
    #I've been trying to do randomize the selection of the videos but can't get it to work, basically just fucking around w trying to
    #code random.shuffle ??? anyways i took it out bc otherwise it'll break. hopefully u see the vision
    
    # Write when it's initialized
    resdict['Timepoint'], resdict['Time'] = 'Movie Init', timer.getTime()
    writer.writerow(resdict)
    resdict['Timepoint'], resdict['Time'] = None,None
    
    # Create two different lists of videos for trial 1 and trial 2. 
    trialvideo = os.path.join(os.getcwd(),"taskScripts",filename[1])
    trialsplits = pd.read_csv(os.path.join(os.getcwd(),"taskScripts",filename[0]))
    #trialvideo = os.path.join(os.getcwd(), 'taskScripts//resources//Movie_Task//videos') + "/" + list_of_videos[filename-1]
    #trialsplits = pd.read_csv(os.path.join(os.getcwd(), 'taskScripts//resources//Movie_Task//csv//probetimes_orders.csv'))
    videoname = filename[1].rsplit("/",1)[-1]
    trialname = "Movie Task-" + trialvideo.split(".")[0].split("/")[-1]
    vern = probever
    trialsplit = trialsplits.iloc[vern]
    
    # Pick the video to show based on the trial version, we are just going to pick the one at the top of the list
    
        
    
    
    # present film using moviestim
    resdict['Timepoint'], resdict['Time'],resdict['Auxillary Data'] = 'Movie Start', timer.getTime(), videoname
    writer.writerow(resdict)
    resdict['Timepoint'], resdict['Time'],resdict['Auxillary Data'] = None,None,None
    
    text_inst = visual.TextStim(win=win, name='text_1',
                        text='Loading...',
                        font='Open Sans',
                        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
                        color='black', colorSpace='rgb', opacity=None, 
                        languageStyle='LTR',
                        depth=0.0)
    text_inst.draw()
    win.flip()
     
    mov = visual.MovieStim3(win, trialvideo, size=(1920, 1080), flipVert=False, flipHoriz=False, loop=False)


    expClock = core.Clock()

    boundary_times = [] #initialize event segmentation list
    
    timelimit = trialsplit[0]
    #trialsplit = trialsplit
    trialsplit = trialsplit.diff()[1:]
    esqshown = False
    resettime = True
    en = 0
    timelimitpercent = int(100*(timelimit/runtime))
    while mov.status != visual.FINISHED:
        if expClock.getTime() < runtime:
            time = expClock.getTime()
            if expClock.getTime() > timelimit:

                try:
                    timelimit = trialsplit.values[en]
                except:
                    timelimit = 10000
                    pass
                en += 1
                mov.pause()
                timepause = runtime - expClock.getTime()
                ESQ.runexp(None,timer,win,[writer,writera],resdict,None,None,None,movietype=trialname) 
                text_inst.draw()
                win.flip()
                #mov.draw()
                writera.writerow({'Timepoint':'EXPERIMENT DATA:','Time':'Experience Sampling Questions'})
                writera.writerow({'Timepoint':'Start Time','Time':timer.getTime()})
                resdict['Assoc Task'] = None
                resdict['Timepoint'], resdict['Time'],resdict['Auxillary Data'] = 'Movie prompt {} {}'.format(en,videoname), timer.getTime(), timelimitpercent
                writer.writerow(resdict)
                resdict['Timepoint'], resdict['Time'],resdict['Auxillary Data'] = None,None,None
                #win.flip()
                mov.play()
                resettime = True
                
                #runtime = timepause
                esqshown = True
                #break
            if resettime:
                expClock.reset()
                resettime = False
            mov.draw()
            win.flip()
            keys = event.getKeys(timeStamped=expClock)
            for key, timestamp in keys:
                if key == 'space':
                    boundary_times.append(round(timestamp, 3))
        else:
            break

    #at the end of each clip, present comprehension questions
    if filename[1] == "resources/Movie_Task/videos/friends1.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 1, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 2, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 3, participant_id, videoname, responses_data)
        #responses_data = present_comprehension_question(win, stim, 4, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 13, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
        

    if filename[1] == "resources/Movie_Task/videos/friends2.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 4, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 5, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 6, participant_id, videoname, responses_data)
        #responses_data = present_comprehension_question(win, stim, 8, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 14, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    if filename[1] == "resources/Movie_Task/videos/friends3.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 7, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 8, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 9, participant_id, videoname, responses_data)
        #responses_data = present_comprehension_question(win, stim, 12, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 15, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    if filename[1] == "resources/Movie_Task/videos/friends4.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        responses_data = present_comprehension_question(win, stim, 10, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 11, participant_id, videoname, responses_data)
        responses_data = present_comprehension_question(win, stim, 12, participant_id, videoname, responses_data)
        #responses_data = present_comprehension_question(win, stim, 16, participant_id, videoname, responses_data)
        save_comp_csv(responses_data, participant_id, clipname, seed)
        seen_data = present_seen_question(win, stim, 16, participant_id, videoname, seen_data)
        save_seen_csv(seen_data, participant_id, clipname, seed)
    
    # Save event segmentation data
    event_seg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "event_seg")
    os.makedirs(event_seg_dir, exist_ok=True)

    if filename[1] == "resources/Movie_Task/videos/test1.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        # Save event boundary data
        save_eventseg_csv(boundary_times, participant_id, clipname, seed)

    if filename[1] == "resources/Movie_Task/videos/test2.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        save_eventseg_csv(boundary_times, participant_id, clipname, seed)

    if filename[1] == "resources/Movie_Task/videos/friends3.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        save_eventseg_csv(boundary_times, participant_id, clipname, seed)

    if filename[1] == "resources/Movie_Task/videos/friends4.mp4":
        base_name = os.path.splitext(os.path.basename(filename[1]))[0]
        clipname = base_name.split('.')[0]
        save_eventseg_csv(boundary_times, participant_id, clipname, seed)



    



    
    
    
    
    
    return trialname


#function to run the segmentation practice
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
                    core.wait(1)  # prevent high CPU usage



