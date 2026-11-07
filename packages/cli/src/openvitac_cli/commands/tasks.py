import cyclopts

from openvitac_tasks import TASKS, TaskName

tasks = cyclopts.App(name="tasks", help="OpenViTac Prefabricated Tasks")


@tasks.command
def list_tasks():
    """
    List all tasks
    """
    for name in TASKS:
        print(name)


tasks.default_command = list_tasks


@tasks.command
def show(name: TaskName):
    """
    Show details about a specific task
    """
    print(TASKS[name])
