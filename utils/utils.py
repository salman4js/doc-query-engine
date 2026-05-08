def show_help_commands():
    print("Please pass --read, --store, --delete or --answer")
    print("Example: python main.py --store --file report.pdf")
    print("--read: Gives all the colletion's chunks from the database")
    print("--store: Expects a file with '--file' command. Performs chunking, embedding of the file passed and store the results to the database")
    print("--delete: Deletes all the collection's data from the database (Wipes out everything)")
    print("--answer: Expects a question from the user with '--question'")
    print("--model_name: Expects a model name from the user with '--model_name'")


def show_no_arguments_command():
    print("No arguments has been passed, run 'python3 main.py --help' to see available commands")

def show_no_model_msg():
    print("Model name must be mandatory to generate answers")
    print("Example: python3 main.py --answer --model_name 'model_name' --question 'your_question_here'")
    print('Run python3 main.py --help for more information')