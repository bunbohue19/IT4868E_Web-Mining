import mteb

# Get tasks in multiple languages
multilingual_tasks = mteb.get_tasks(
    languages=["jpn"]
)

for task in multilingual_tasks:
    print(task)