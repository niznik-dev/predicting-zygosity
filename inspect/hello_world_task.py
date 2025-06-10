from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message

@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="Just reply with Hello World",
                target="Hello World",
            ),
            Sample(
                input="Say Goodbye",
                target="Goodbye",
            ),
        ],
        solver=[generate()],
        scorer=exact(),
    )


data=[
    Sample(
        input="Just reply with Hello World",
        target="Hello World",
        ),
        Sample(
            input="Say Goodbye",
            target="Goodbye",
        )
]


@task
def hello_world_local_no_prompt():
    return Task(
        dataset=data,
        solver=[generate()],
        scorer=exact(),
        model="hf//scratch/gpfs/mjs3/llm_base_models/Llama-3.2-1B-Instruct"
    )


@task
def hello_world_local_helpful_prompt_1B():
    return Task(
        dataset=data,
        solver=[
            system_message("This is a test, and you will be evaluated based on how well you follow directions. Please read the instructions carefully respond with the minimal neccesary response."), 
            generate()
        ],
        scorer=exact(),
        model="hf//scratch/gpfs/mjs3/llm_base_models/Llama-3.2-1B-Instruct"
    )

@task
def hello_world_local_helpful_prompt_8B():
    return Task(
        dataset=data,
        solver=[
            system_message("This is a test, and you will be evaluated based on how well you follow directions. Please read the instructions carefully respond with the minimal neccesary response."), 
            generate()
        ],
        scorer=exact(),
        model="hf//scratch/gpfs/mjs3/llm_base_models/Meta-Llama-3.1-8B-Instruct"
    )


######### time to generalize

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message

# Define your test data once
data = [
    Sample(
        input="Just reply with Hello World",
        target="Hello World",
    ),
    Sample(
        input="Say Goodbye",
        target="Goodbye",
    )
]

# Define your models
models = {
    "llama_1b": "hf//scratch/gpfs/mjs3/llm_base_models/Llama-3.2-1B-Instruct",
    "llama_8b": "hf//scratch/gpfs/mjs3/llm_base_models/Meta-Llama-3.1-8B-Instruct",
    # Add more models here as needed
}

# Helper function to create tasks
def create_hello_world_task(model_path, with_prompt=False):
    solvers = [generate()]
    if with_prompt:
        solvers.insert(0, system_message(
            "This is a test, and you will be evaluated based on how well you follow directions. "
            "Please read the instructions carefully respond with the minimal necessary response."
        ))
    
    return Task(
        dataset=data,
        solver=solvers,
        scorer=exact(),
        model=model_path
    )

# Generate tasks for each model
@task
def hello_world_1b_no_prompt():
    return create_hello_world_task(models["llama_1b"])

@task
def hello_world_1b_with_prompt():
    return create_hello_world_task(models["llama_1b"], with_prompt=True)

@task
def hello_world_8b_no_prompt():
    return create_hello_world_task(models["llama_8b"])

@task
def hello_world_8b_with_prompt():
    return create_hello_world_task(models["llama_8b"], with_prompt=True)