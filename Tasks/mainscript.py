
# Main script written by Ian Goodall-Halliwell. Subscripts are individually credited. Many have been extensively modified, for better or for worse (probably for worse o__o ).

from psychopy import visual 
from psychopy import core, gui, event

import time
import csv

import pandas as pd

import yaml
from taskScripts import ESQ, movieTask, eventSegmentation
from taskScripts import freeassociationTask as freeAssoc
import os
import random

from taskScripts import Lingering_Question as lingerQuestion

os.chdir(os.path.dirname(os.path.realpath(__file__)))
if not os.path.exists(os.path.join(os.getcwd(),"log_file")):
        os.mkdir(os.path.join(os.getcwd(),"log_file"))
# This class is responsible for creating and holding the information about how each task should run.
# It contains the number of repetitions and a global runtime variable.
# It also contains the subject ID, and will eventually use the experiment seed to randomize trial order.
class metadatacollection():
        def __init__(self,INFO):#, main_log_location):
                self.INFO = INFO
                #self.main_log_location = main_log_location

                # Don't really know what this is, best to leave it be probably
                self.sbINFO = "Test"


        # This opens the GUI        
        def rungui(self):
                self.sbINFO = gui.DlgFromDict(self.INFO)

        # This writes info collected from the GUI into the logfile
        def collect_metadata(self):  
                print(self.sbINFO.data)
                if os.path.exists(os.path.join(os.getcwd()+self.sbINFO.data[-1])): 
                        os.remove(os.path.join(os.getcwd()+self.sbINFO.data[-1]))
                if not os.path.exists(os.path.join(os.getcwd(),'log_file')):
                        os.mkdir(os.path.join(os.getcwd(),'log_file'))
                f = open(os.path.join(os.getcwd() + '/log_file/output_log_{}_{}_full.csv'.format(self.sbINFO.data[-1],self.INFO['Experiment Seed'])), 'w', newline='')
                fq = open(os.path.join(os.getcwd() + '/log_file/output_log_{}_{}.csv'.format(self.sbINFO.data[-1],self.INFO['Experiment Seed'])), 'w', newline='')
                metawriter = csv.writer(f)
                metawriter2 = csv.writer(fq)
                metawriter.writerow(["METADATA:"])
                metawriter.writerow(self.sbINFO.inputFieldNames)
                metawriter.writerow(self.sbINFO.data)
                metawriter.writerow(["Index", "Field Name", "Value"])
                for idx, name in enumerate(self.sbINFO.inputFieldNames):
                        metawriter.writerow([idx, name, self.sbINFO.data[idx]])
                metawriter2.writerow(["METADATA:"])
                metawriter2.writerow(self.sbINFO.inputFieldNames)
                metawriter2.writerow(self.sbINFO.data)
                random.seed(a=int(metacoll.INFO['Experiment Seed']))
                metawriter.writerow(["Time after setup", taskbattery.time.getTime()])
                metawriter2.writerow(["Time after setup",taskbattery.time.getTime()])
                writer = csv.DictWriter(f, fieldnames=taskbattery.resultdict)
                writer.writeheader()
                writer2 = csv.DictWriter(fq, fieldnames=taskbattery.resultdict)
                writer2.writeheader()
                f.close()
                fq.close()
        
        
# Creates a list of all the tasks, and allows you to iterate through them without closing the window
class taskbattery(metadatacollection):
        time = core.Clock()
        resultdict = {'Timepoint': None, 'Time': None, 'Is_correct': None, 'Experience Sampling Question': None, 'Experience Sampling Response':None, 'Task' : None, 'Task Iteration': None, 'Participant ID': None, 'Response_Key':None, 'Auxillary Data': None, 'Assoc Task':None}
        #self.win = visual.Window(size=(1280, 800),color='white', winType='pyglet',fullscr=True)
        def __init__(self, tasklist, ESQtask, INFO):
                self.tasklist = tasklist
                taskbattery.ESQtask = ESQtask
                self.INFO = INFO
                self.taskexeclist = []
                self.win = visual.Window(size=(1440, 960),color='white',fullscr=False)
                self.text = text_2 = visual.TextStim(win=self.win, name='text_2',
                        text='Welcome to our experiment. \n Please follow the instructions on-screen and notify the attending researcher if anything is unclear \n We are thankful for your participation. \n Press <return/enter> to continue.',
                        font='Arial',
                        anchorHoriz='center', anchorVert='center', wrapWidth=None, ori=0, 
                        color='black', colorSpace='rgb', opacity=1, 
                        languageStyle='LTR',
                        depth=0.0)
                taskbattery.win = self.win
        
        
        #def initializeBattery(self):
                #for i in self.tasklist:

        def run_battery(self):
                
                
                for en,i in enumerate(self.tasklist):
                        os.chdir(os.getcwd())
                        i.show()
                        i.run()
                        pp = len(self.tasklist)
                        #if en < len(self.tasklist):
                                #i.end()
                        
                        
#OPEN THE TRIAL FILES AND CUT THEM INTO BLOCKS

# This creates a class which feeds all the necessary information into the task functions imported from each task file
# Allows you to create different task instances, which will be useful for creating blocks (my current project)
# Saves the log file after each task. It takes some extra time, but it prevents a crash from corrupting the file
class task(taskbattery,metadatacollection):
        def __init__(self, task_module, main_log_location, backup_log_location,  name, trialclass, runtime, dfile, ver, probever=None,esq=False):#, trialfile, ver):
                self.main_log_location = main_log_location
                self.backup_log_location = backup_log_location # Not yet implemented
                self.task_module = task_module # The imported task function
                self.name = name # A name for each task to be written in the logfile
                self.probever = probever
                self.trialclass = trialclass # Has something to do with writing task name into the logfile I think? Probably don't touch this.
                self.runtime = runtime # A "universal" "maximum" time each task can take. Will not stop mid trial, but will prevent trial repetions after the set time in seconds
                self.esq = esq
                self.ver = ver
                self.dfile = dfile
                #super.__init__()
        
        def initvers(self):
                os.chdir(os.getcwd())
                try:
                        f = open(os.path.join(os.getcwd(), self.dfile), 'r', newline="")
                except:
                        try:
                                f = open(os.path.join(os.path.join(os.getcwd(),'taskScripts'), self.dfile), 'r', newline="")
                        except:
                                f = open(os.path.join(os.getcwd(), self.dfile), 'r', newline="")
                d_reader = csv.DictReader(f)

                #get fieldnames from DictReader object and store in list
                self.headers = d_reader.fieldnames
                r = csv.reader(f)
                l = []
                for row in r:
                        l.append(row)
                        
                self.l = l
                
                incrordecr = random.choice([-1,1])
                amnt = random.randint(5,15)
                self.runtime = self.runtime + amnt*incrordecr
                with open(self.main_log_location, 'a', newline="") as o:
                        metawrite = csv.writer(o)
                        metawrite.writerow(" ")
                        metawrite.writerow(["Runtime Mod",(amnt*incrordecr)])
        def setver(self):
                ###
                ###   ###   NEED TO LOAD THE FILES FROM CSV AND PRESERVE THE HEADERS, THEN PUT THE OLD HEADERS INTO THE NEW BLOCK CSVs
                ###
                self.ver_a = random.sample(self.l, round(len(self.l)/2) )
                b = self.l
                for val in self.ver_a:
                        b[:] = [x for x in b if x != val]
                random.shuffle(b)
                b.insert(0,*[self.headers])
                self.ver_b = b
                random.shuffle(self.ver_a)
                
                self.ver_a.insert(0, self.headers)
                
                lis = [self.ver_a,self.ver_b]
                for enum, thing in enumerate(lis):
                        
                        if not os.path.exists(os.getcwd()+ '//tmp'):
                                os.mkdir(os.getcwd()+ '//tmp')
                        if not os.path.exists(os.getcwd()+ '//tmp//%s' %(self.name)):
                                os.mkdir(os.getcwd() + '//tmp//%s' %(self.name))
                        if os.path.exists(os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(enum) + ".csv")):
                                os.remove(os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(enum) + ".csv"))
                        with open(os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(enum) + ".csv"), mode='w', newline="") as file:
                                
                                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                for subthing in thing:
                                        file_writer.writerow(subthing)
                self._ver_a_name = os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(0) + ".csv")
                self._ver_b_name = os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(1) + ".csv")
        def setver3(self):
                ###
                ###   ###   NEED TO LOAD THE FILES FROM CSV AND PRESERVE THE HEADERS, THEN PUT THE OLD HEADERS INTO THE NEW BLOCK CSVs
                ###
                self.ver_a = random.sample(self.l, round(len(self.l)/3) )
                b = self.l
                for val in self.ver_a:
                        b[:] = [x for x in b if x != val]
                random.shuffle(b)
                
                self.ver_b = random.sample(b, round(len(b)/2) )
                c = b
                for val in self.ver_a:
                        c[:] = [x for x in c if x != val]
                random.shuffle(c)
                
                
                c.insert(0,*[self.headers])
                self.ver_c = c
                
                random.shuffle(self.ver_a)
                random.shuffle(self.ver_b)
                
                self.ver_a.insert(0, self.headers)
                self.ver_b.insert(0, self.headers)
                
                lis = [self.ver_a,self.ver_b,self.ver_c]
                for enum, thing in enumerate(lis):
                        
                        if not os.path.exists(os.getcwd()+ '//tmp'):
                                os.mkdir(os.getcwd()+ '//tmp')
                        if not os.path.exists(os.getcwd()+ '//tmp//%s' %(self.name)):
                                os.mkdir(os.getcwd() + '//tmp//%s' %(self.name))
                        if os.path.exists(os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(enum) + ".csv")):
                                os.remove(os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(enum) + ".csv"))
                        with open(os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(enum) + ".csv"), mode='w', newline="") as file:
                                
                                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                for subthing in thing:
                                        file_writer.writerow(subthing)
                self._ver_a_name = os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(0) + ".csv")
                self._ver_b_name = os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(1) + ".csv")   
                self._ver_c_name = os.getcwd() + '//tmp//%s' %(self.name + "//" + self.name + "_version_"+ str(2) + ".csv")        
                self._ver_d_name = None         
                                
        def run(self):
                global prevname
                if not os.path.exists(self.main_log_location):
                        if self.main_log_location.split(".")[1] == None:
                                os.mkdir(self.main_log_location.split("/")[0])      
                fr = open(self.main_log_location, 'a', newline="")
                f = open((self.main_log_location.split(".")[0] + "_full.csv"), 'a', newline="")
                fre = csv.writer(fr)
                r = csv.writer(f)
                
                writer = csv.DictWriter(f, fieldnames=taskbattery.resultdict)
                r.writerow(["EXPERIMENT DATA:",self.name])
                r.writerow(["Start Time", taskbattery.time.getTime()])
                writer2 = csv.DictWriter(fr, fieldnames=taskbattery.resultdict)
                fre.writerow(["EXPERIMENT DATA:",self.name])
                fre.writerow(["Start Time", taskbattery.time.getTime()])
                taskbattery.resultdict = {'Timepoint': None, 'Time': None, 'Is_correct': None, 'Experience Sampling Question': None, 'Experience Sampling Response':None, 'Task' : self.name, 'Task Iteration': '1', 'Participant ID': self.trialclass[1], 'Response_Key' : None, 'Auxillary Data': None, 'Assoc Task':None}
                if self.esq == False:
                        if self.ver == 1:
                                dataver = self.task_module.runexp(self.backup_log_location, taskbattery.time, taskbattery.win, [writer,writer2], taskbattery.resultdict, self.runtime, self._ver_a_name, int(metacoll.INFO['Experiment Seed']),self.probever,self.trialclass[1])
                        if self.ver == 2:
                                dataver = self.task_module.runexp(self.backup_log_location, taskbattery.time, taskbattery.win, [writer,writer2], taskbattery.resultdict, self.runtime, self._ver_b_name, int(metacoll.INFO['Experiment Seed']),self.probever,self.trialclass[1])
                        if self.ver == 3:
                                dataver = self.task_module.runexp(self.backup_log_location, taskbattery.time, taskbattery.win, [writer,writer2], taskbattery.resultdict, self.runtime, self._ver_c_name, int(metacoll.INFO['Experiment Seed']),self.probever,self.trialclass[1])
                        if self.ver == 4:
                                dataver = self.task_module.runexp(self.backup_log_location, taskbattery.time, taskbattery.win, [writer,writer2], taskbattery.resultdict, self.runtime, self._ver_d_name, int(metacoll.INFO['Experiment Seed']),self.probever,self.trialclass[1])
                
                if self.esq == True:
                        
                        taskbattery.resultdict = {'Timepoint': None, 'Time': None, 'Is_correct': None, 'Experience Sampling Question': None, 'Experience Sampling Response':None, 'Task' : self.name, 'Task Iteration': '1', 'Participant ID': self.trialclass[1], 'Response_Key' : None, 'Auxillary Data': None, 'Assoc Task':taskbattery.prevname}
                        self.task_module.runexp(self.backup_log_location, taskbattery.time, taskbattery.win, [writer,writer2], taskbattery.resultdict, self.runtime, None, int(metacoll.INFO['Experiment Seed']))
                        dataver = None
                f.close()
                fr.close()
                taskbattery.resultdict = {'Timepoint': None, 'Time': None, 'Is_correct': None, 'Experience Sampling Question': None, 'Experience Sampling Response':None, 'Task' : None, 'Task Iteration': None, 'Participant ID': None,'Response_Key':None, 'Auxillary Data': None, 'Assoc Task':None}
                if dataver != None:
                        self.name = self.name + "-" + dataver
                taskbattery.prevname = self.name



class taskgroup(taskbattery,metadatacollection):
        def __init__(self,tasks,instrpath):
                self.tasks = tasks
                self.instrpath = instrpath
                
                
                
        def show(self):
                text_inst = visual.TextStim(win=taskbattery.win, name='text_4',
                        text='',
                        font='Open Sans',
                        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
                        color='black', colorSpace='rgb', opacity=1.0, 
                        languageStyle='LTR',
                        depth=0.0)
                try:
                        with open(os.path.join(os.path.join(os.getcwd(),"taskScripts"),self.instrpath), newline="") as f:
                                lines1 = f.read()
                except:
                        with open(os.path.join(os.getcwd(),self.instrpath), newline="") as f:
                                lines1 = f.read()
                text_inst.setText(lines1)
                text_inst.draw()
                taskbattery.win.flip()
                time.sleep(1)
                event.waitKeys(keyList=['return'])
                
        def run(self):
                for taskgrp in self.tasks:
                        if taskgrp == None:
                                continue
                        for task in taskgrp:
                                print("Now initializing {}".format(task.name))
                                task.initvers()
                                print("Now setting up {}".format(task.name))
                                task.setver3()
                                print("Now running {}".format(task.name))
                                task.run()
                                #print("Now starting ESQ for {}".format(task.name))
                                #taskbattery.ESQtask.run()
                        
        def end(self):
                text_inst = visual.TextStim(win=taskbattery.win, name='text_1',
                        text='This is the end of the experiment. \n \n Please inform the attending researcher you have completed testing. \n \n Thank you for participating!',
                        font='Open Sans',
                        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
                        color='black', colorSpace='rgb', opacity=None, 
                        languageStyle='LTR',
                        depth=0.0)
                text_inst.draw()
                taskbattery.win.flip()
                time.sleep(1)
                event.waitKeys(keyList=['return'])
                taskbattery.win.flip()
                
        #def shuffle(self):
                #a = self.tasks
                #for a in self.tasks:
                        #random.shuffle(a)
                #random.shuffle(self.tasks)
                #print('')
                        
        
        

### HAVE TO EXPOSE ESQ TASK TO MOVIE SCRIPT

if __name__ == "__main__":
        
        with open(os.path.join(os.pardir, 'config.yaml'),'r') as f:
                config = yaml.safe_load(f)
        # Info Dict
        INFO = {
                        'Experiment Seed': random.randint(1, 9999999),  
                        'Subject': 'Enter ID Here', 
                        #"Probe 1 Version":"",
                        #"Probe 2 Version":"",
                        #"Probe 3 Version":"",
                        #"Probe 4 Version":"",
                        
                }



        # Main and backup data file


        # Run the GUI and save output to logfile
        metacoll = metadatacollection(INFO)
        metacoll.rungui()

        #probeorders = pd.read_csv("taskScripts/resources/Movie_Task/csv/counterbalanced_orders_n120.csv")
        #probeversions = 90

        #if int(metacoll.INFO['Subject'])>probeversions:
                #probeiter = int(metacoll.INFO['Subject'])//probeversions
                #sub_probe = int(metacoll.INFO['Subject']) - (probeversions*probeiter)
        #else: 
                #sub_probe = int(metacoll.INFO['Subject'])

        #probe1_version = probeorders.loc[probeorders['participant_number'] == sub_probe, 'Clip 1'].values
        #probe2_version = probeorders.loc[probeorders['participant_number'] == sub_probe, 'Clip 2'].values
        #probe3_version = probeorders.loc[probeorders['participant_number'] == sub_probe, 'Clip 3'].values
        #probe4_version = probeorders.loc[probeorders['participant_number'] == sub_probe, 'Clip 4'].values

        #probe1_version = probe1_version[0]
        #probe2_version = probe2_version[0]
        #probe3_version = probe3_version[0]
        #probe4_version = probe4_version[0]

        metacoll.collect_metadata()
        metacoll.INFO['Block Runtime'] = 48000
        
        # Defining output datafile
        datafile = str(os.getcwd() + '/log_file/output_log_{}_{}.csv'.format(metacoll.INFO['Subject'],metacoll.INFO['Experiment Seed']))
        datafileBackup = 'log_file/testfullbackup.csv'
        if not os.path.exists("tmp"):
                os.mkdir("tmp")
        # with open("tmp/esqtmp.pkl",'wb') as frrr:
        #         pkl.dump([datafile,datafileBackup,metacoll.sbINFO.data,int(metacoll.INFO['Block Runtime'])],frrr)
        #ESQTask = task(ESQ, datafile, datafileBackup, "Experience Sampling Questions", metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources/GoNoGo_Task/gonogo_stimuli.csv',1, esq=True)
        
        ESQTask = None

        #Select a random int between 1 and 10
        #random_probe_version = random.randint(1,10)

        # Defining each task as a task object
        movieTask1 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/lms.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 1,1)
        movieTask2 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/12_years.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 2,1)
        movieTask3 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/500Days.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 3,1)
        movieTask4 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/backToFuture.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 4,1)
        movieTask5 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/c4.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 4,1)
        movieTask6 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/prestige.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 4,1)
        movieTask7 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/pulpFiction.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 4,1)
        movieTask8 = task(eventSegmentation, datafile, ["resources/Movie_Task/csv/probetimes_orders.csv","resources/Movie_Task/videos/shawshank.mp4"],"Event Segmentation Task",  metacoll.sbINFO.data, int(metacoll.INFO['Block Runtime']),'resources//Movie_Task//csv//sorted_filmList.csv', 4,1)

        #moviegroup = [movieTask1,movieTask2,movieTask3]

        moviegroup = [movieTask1, movieTask2, movieTask3, movieTask4, movieTask5, movieTask6, movieTask7, movieTask8]
        

        cue_word_map = {
                "lms.mp4": ("ROADTRIP", "ICE-CREAM"),
                "12_years.mp4": ("FREEDOM", "MASTER"),
                "500Days.mp4": ("LOVE", "ROMANCE"),
                "backToFuture.mp4": ("TIME", "PAST"),
                "c4.mp4": ("SURVEILLANCE", "GOVERNMENT"),
                "prestige.mp4": ("MAGIC", "BREAKUP"),
                "pulpFiction.mp4": ("BRIEFCASE", "ROBBERY"),
                "shawshank.mp4": ("PRISON", "ESCAPE")
         }
        
        random.shuffle(moviegroup) #just trying to shuffle the video clips 

        print(moviegroup)

        
        

        ## Determine what the first video in the random shuffle is - to be used for free association task
        first_movie = moviegroup[0].backup_log_location[1]  # second element in the list
        first_video_name = os.path.basename(first_movie)  
        
        movie_cues = list(cue_word_map[first_video_name])
        random.shuffle(movie_cues)
        cue_pre = movie_cues[0]
        cue_post = movie_cues[1]

        print(f"First movie: {first_video_name}")
        print(f"PreFA cue: {cue_pre} | PostFA cue: {cue_post}")

        


        movie_main = taskgroup([moviegroup],"resources/group_inst/movie_main.txt")


        fulltasklist = [movie_main]
        
        if config['randomize_task']:
                fulltasklist = random.sample(list(fulltasklist),config['num_task_in_subset'])
        
        
        # Shuffles the order of the tasks in taskgroups
        #for enum, blk in enumerate(fulltasklist):
               # blk.shuffle()
        
        # Shuffle the order of the taskgroups
        #random.shuffle(fulltasklist)
        tasks = fulltasklist

        tbt = taskbattery(tasks, ESQTask, INFO)

        tbt.text.draw()
        tbt.win.flip()
        time.sleep(1)
        event.waitKeys(keyList=['return'])
        tbt.win.flip()

        #eventSegmentation.run_practice(tbt.win) #to run the practice before the battery

        # === PreFA ===
        print("Starting Free Association Task (before movies)...")
        freeAssoc.show_instructions(tbt.win)
        responses_pre = freeAssoc.runexp(
                tbt.win,
                participant_id=metacoll.INFO['Subject'],
                cue_word=cue_pre,
                duration=300,  # 5 min
                VideoName=first_video_name
        )
        freeAssoc.save_association(
                responses_pre,
                participant_id=metacoll.INFO['Subject'],
                label="_PRE",
                VideoName=first_video_name
        )

        # Run movies
        tbt.run_battery()

        # === PostFA ===
        print("Starting Free Association Task (after movies)...")
        freeAssoc.show_instructions(tbt.win)
        responses_post = freeAssoc.runexp(
                tbt.win,
                participant_id=metacoll.INFO['Subject'],
                cue_word=cue_post,
                duration=300,  # 5 min
                VideoName=first_video_name
        )
        freeAssoc.save_association(
                responses_post,
                participant_id=metacoll.INFO['Subject'],
                label="_POST",
                VideoName = first_video_name
        )

        # After final video & final free association

        print("Starting Lingering Self-Report (after movies)...")
        lingerQuestion.show_instructions(tbt.win)

        response_linger = lingerQuestion.runexp(
                tbt.win,
                participant_id=metacoll.INFO['Subject'],
                VideoName=first_video_name
        )

        lingerQuestion.save_response(
                response_linger,
                participant_id=metacoll.INFO['Subject'],
                VideoName=first_video_name
        )

        movie_main.end()

        print("Success")