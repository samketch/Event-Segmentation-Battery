from psychopy import visual, event, core
import pandas as pd
import os

def load_comprehension_questions(episode_number):
    """
    Loads comprehension questions for the given episode from a CSV file.
    """
    csv_path = os.path.join(os.getcwd(), "taskScripts/resources/Movie_Task/csv/comprehension_questions.csv")

    # Debugging: Print path
    print(f"Loading comprehension questions from: {csv_path}")

    # Ensure the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Read the CSV safely
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Error: CSV file is empty!")
    except pd.errors.EmptyDataError:
        raise ValueError("Error: CSV file contains no data or has a format issue!")

    print("Loaded questions:", df.head())  # Debugging print

    # Filter questions for the given episode
    episode_questions = df[df['episode'] == episode_number]

    # Convert to dictionary format
    questions = []
    for _, row in episode_questions.iterrows():
        questions.append({
            "question": row['question'],
            "choices": [row['choice_1'], row['choice_2'], row['choice_3'], row['choice_4']],
            "correct": row['correct_answer']
        })

    return questions

def runexp(win, writer, resdict, episode_number):
    """
    Runs the comprehension questions task for the given episode.
    """
    questions = load_comprehension_questions(episode_number)
    responses = []

    for i, q in enumerate(questions):
        # Display question
        question_text = f"{q['question']}\n\n"
        for idx, choice in enumerate(q['choices']):
            question_text += f"({idx + 1}) {choice}\n"
        
        stim = visual.TextStim(win, text=question_text, color=[-1, -1, -1], wrapWidth=1300, units="pix", height=40)
        stim.draw()
        win.flip()
        
        # Wait for user response (1-4)
        response = event.waitKeys(keyList=['1', '2', '3', '4'], timeStamped=True)
        selected_choice = q['choices'][int(response[0][0]) - 1]  # Convert response index to choice text
        is_correct = selected_choice == q["correct"]

        # Log response
        resdict['Timepoint'], resdict['Time'] = f'Comprehension Q{i+1}', core.getTime()
        resdict['Auxillary Data'] = f'Q: {q["question"]}, Response: {selected_choice}, Correct: {is_correct}'
        writer.writerow(resdict)

        responses.append((q["question"], selected_choice, is_correct))

    return responses
