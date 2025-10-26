# free_association_task_equivalent.py
# Python/PsychoPy equivalent of EvoPsyc (jsPsych) free association task

import os, csv
from datetime import datetime
from psychopy import visual, event, core

# Define save directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "free_association")
os.makedirs(SAVE_DIR, exist_ok=True)



def show_instructions(win):
    """Show instructions just like EvoPsyc version."""
    instructions = (
        "Word Chain Game\n\n"
        "Please read the following instructions carefully.\n\n"
        "Type any words that come to mind, followed by the ENTER key.\n\n"
        "Your response will briefly appear onscreen, adding to your word chain.\n\n"
        "This game will last 5 minutes.\n\n"
        "We are interested in the associations you have between words.\n\n"
        "Please feel free to type any words that come to mind.\n\n"
        "Do not form sentences. Instead, please use individual words to form your word chain.\n\n"
        "Press ENTER to begin."
    )

    stim = visual.TextStim(win, text=instructions, wrapWidth=1300, color=[-1, -1, -1], height=40, units="pix")
    stim.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    win.flip()


def runexp(win, participant_id, VideoName=None, cue_word="TEST", duration=10):
    """
    Free association task:
    - Starts with externally provided cue word (capitalized, center).
    - Cue word shown for 2s, then disappears.
    - Participant types associate (visible in lowercase where typed).
    - Typed word flashes for 250 ms (same position, same case).
    - Then shown capitalized in center for 500 ms as new cue.
    """
    responses = []
    clock = core.Clock()
    clock.reset()

    # Prepare text objects
    cue_text = visual.TextStim(win, text="", height=0.1, color=[-1, -1, -1], pos=(0, 0.0))
    input_text = visual.TextStim(win, text="", height=0.08, color=[-1, -1, -1], pos=(0, -0.2))

    while clock.getTime() < duration:
        # === Step 1: show cue word (CAPS, center) for 2s ===
        if clock.getTime() <= 2:
            cue_text.text = cue_word.upper()
            cue_text.pos = (0, 0.0)
            cue_text.draw()
            win.flip()
            core.wait(2.0)
        else: 
            cue_text.text = cue_word.upper()
            cue_text.pos = (0, 0.0)
            cue_text.draw()
            win.flip()
            core.wait(0.5)

        # clear screen after cue
        win.flip()

        # === Step 2: collect response (typed input, lower case, visible as typed) ===
        typed_word = ""
        response_clock = core.Clock()
        new_word = None

        while new_word is None and clock.getTime() < duration:
            keys = event.getKeys(timeStamped=response_clock)
            for key, t in keys:
                if key == "return" and typed_word.strip() != "":
                    new_word = typed_word.strip()
                    responses.append((participant_id, VideoName, cue_word.upper(), new_word, t))
                    cue_word = new_word
                    # save (cue, response, RT)
                    core.wait(0.25)
                    break

                elif key == "backspace":
                    typed_word = typed_word[:-1]
                elif key in ["escape"]:
                    win.close()
                    core.quit()
                else:
                    if len(key) == 1:
                        typed_word += key

            # show current typed word during input
            input_text.text = typed_word
            input_text.pos = (0, -0.2)
            input_text.draw()
            win.flip()

    return responses



def save_association(responses, participant_id, VideoName=None, label=""):
    """
    Saves free association responses to a CSV in the fixed SAVE_DIR.

    Parameters:
        responses : list of (cue_word, response, RT) tuples
        participant_id : str
        label : str (e.g., "_PRE" or "_POST")
        video_name : str (e.g., "prestige.mp4") — logged in output file
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{participant_id}_freeAssoc_{current_datetime}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "VideoName", "CueWord", "Response", "RT (s)"])
        writer.writerows(responses)

    print(f"[SAVED] Free association data saved to {filepath}")
    return filepath


# Standalone test
if __name__ == "__main__":
    win = visual.Window(size=(1440, 960), color="white", fullscr=False)
    show_instructions(win)
    data = runexp(win, participant_id="PTEST", duration=1)  # test 1 min
    save_association(data, participant_id="PTEST")
    win.close()
    core.quit()
