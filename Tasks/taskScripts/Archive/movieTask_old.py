#Written by Brontë McKeown and Theodoros Karapanagiotidis
from psychopy import visual 
import psychopy
psychopy.prefs.hardware['audioLib'] = ['PTB', 'pyo','pygame']
from matplotlib.pyplot import pause

import pandas as pd
from psychopy import gui, data, core,event
from taskScripts import ESQ
import os.path
from taskScripts import comprehensionTask

import random

###################################################################################################
def runexp(filename, timer, win, writer, resdict, runtime,dfile,seed,probever):
    writera = writer[1]
    writer = writer[0]
    random.seed(seed)
    
    resdict['Timepoint'], resdict['Time'] = 'Movie Task Start', timer.getTime()
    writer.writerow(resdict)
    resdict['Timepoint'], resdict['Time'] = None,None
    
    

    

    # user can update instructions for task here if required.
    instructions =      """This experiment will require you to watch a set of 4 episodes of the TV series 'Friends'.

                    \nYou will not be able to pause or rewind. 

                    \nDuring these videos, please listen carefully. There are volume keys on the keyboard to adjust to your liking. You will be asked a series of questions regarding the content of each movie afterwards. 

                    \nPlease do not take any notes during the episodes. 

                    \nPlease do not focus on another window, external devices or attend to other distractions. Do not close this window.

                        """

    # user can update start screen text here if required. 
    start_screen = """Throughout each of the episodes, you will be prompted with questions about your thoughts.

                    \nPlease answer these questions as quickly and honestly as possible. There are no right or wrong answers.

                    \nUse the arrow keys and enter/return key to submit your response. 
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
    stim.setText(start_screen)
    stim.draw()
    win.flip()
    
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
    
    #Trying to use moviestim instead of moviestim3 - NVM - This worked for my mac RIP
    #mov = visual.MovieStim(win, filename=trialvideo, size=(1920, 1080), flipVert=False, flipHoriz=False, loop=False)


    expClock = core.Clock()
    
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
        else:
            break
    # Movie has finished playing, now show comprehension questions

    # Extract episode number from filename (assuming 'friends1.mp4', etc.)
    episode_number = int(videoname.replace("test", "").replace(".mp4", ""))

    # Call comprehension questions task
    comprehensionTask.runexp(win, writer, resdict, episode_number)

    return trialname
