# ==========================================================
# Linger Question Task (Keypress 1–7 Version)
# ==========================================================
# Can be imported into mainscript or run standalone for testing.
#
# Author: Sam Ketcheson
# ==========================================================

from psychopy import visual, event, core
import os, csv

# ----------------------------------------------------------
# Mapping from file names to readable movie titles
# ----------------------------------------------------------
MOVIE_TITLE_MAP = {
    "lms.mp4": "Little Miss Sunshine",
    "500Days.mp4": "500 Days of Summer",
    "c4.mp4": "Citizen Four",
    "shawshank.mp4": "Shawshank Redemption",
    "prestige.mp4": "The Prestige",
    "pulpFiction.mp4": "Pulp Fiction",
    "backToFuture.mp4": "Back to the Future",
    "12_years.mp4": "12 Years a Slave"
}

# ----------------------------------------------------------
# 1. Instructions Screen
# ----------------------------------------------------------
def show_instructions(win):
    """Display brief instructions before the lingering question."""
    instruction_text = visual.TextStim(
        win,
        text="Next, you will answer one final question about the first video.\n\n"
             "Please rate how much the video has lingered in your mind\n"
             "since you watched it.\n\nPress any key to continue.",
        wrapWidth=1.3,
        color=[-1, -1, -1],
        height=0.1
    )
    instruction_text.draw()
    win.flip()
    event.waitKeys()



# ----------------------------------------------------------
# 2. Keypress 1–7 Screen
# ----------------------------------------------------------
def runexp(win, participant_id="TEST", VideoName="test_video"):
    """
    Display a 1–7 numeric choice screen.
    Participant presses a number key (1–7) to respond.
    """
    # Look up the readable title, default to the filename if not found
    movie_title = MOVIE_TITLE_MAP.get(VideoName, VideoName)

    question = visual.TextStim(
        win,
        text=f"To what extent did the first clip you watched ({movie_title}) linger in your mind after watching it?",
        wrapWidth=1.4,
        color=[-1, -1, -1],
        pos=(0, 0.35),
        height=0.1
    )

    scale_low = visual.TextStim(win, text="Not At All", color=[-1, -1, -1], pos=(-0.6, -0.05), height=0.1)
    scale_high = visual.TextStim(win, text="Very Much", color=[-1, -1, -1], pos=(0.6, -0.05), height=0.1)

    numbers = visual.TextStim(
        win,
        text="1     2     3     4     5     6     7",
        color=[-1, -1, -1],
        pos=(0, -0.15),
        height=0.1
    )

    prompt = visual.TextStim(
        win,
        text="Press a number key (1–7) to respond",
        color='gray',
        pos=(0, -0.35),
        height=0.067
    )

    # Draw and wait for response
    for stim in [question, scale_low, scale_high, numbers, prompt]:
        stim.draw()
    win.flip()

    valid_keys = ["1", "2", "3", "4", "5", "6", "7"]
    timer = core.Clock()
    keys = event.waitKeys(keyList=valid_keys + ["escape"], timeStamped=timer)

    key, rt = keys[0]
    if key == "escape":
        core.quit()

    response = int(key)


    return {
        "ParticipantID": participant_id,
        "VideoName": VideoName,
        "Response": response,
        "RT": rt
    }


# ----------------------------------------------------------
# 3. Save Function
# ----------------------------------------------------------
def save_response(response_dict, participant_id, VideoName, save_dir=r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Tasks\lingering_response"):
    """Save the response to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    outfile = os.path.join(save_dir, f"{participant_id}_{VideoName}_linger.csv")

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ParticipantID", "VideoName", "Response", "RT"])
        writer.writerow([
            response_dict["ParticipantID"],
            response_dict["VideoName"],
            response_dict["Response"],
            response_dict["RT"]
        ])

    print(f"Saved lingering response to {outfile}")


# ----------------------------------------------------------
# 4. Standalone Test Mode
# ----------------------------------------------------------
if __name__ == "__main__":
    print("Running Lingering Question (test mode)...")

    win = visual.Window(size=(1440, 960),color='white',fullscr=False)


    show_instructions(win)
    result = runexp(win, participant_id="TEST001", VideoName="friends1")
    save_response(result, participant_id="TEST001", VideoName="friends1")

    thankyou = visual.TextStim(
        win,
        text="Press any key to continue.",
        color=[-1, -1, -1],
        height=0.06
    )
    thankyou.draw()
    win.flip()
    event.waitKeys()

    win.close()
    core.quit()
