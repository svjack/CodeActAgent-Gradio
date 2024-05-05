from typing import List, Optional, Tuple, Dict
History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

import enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


@dataclass(frozen=True)
class Action:
    value: str  # LM returned string for now
    use_tool: bool  # if use_tool == False -> propose answer
    error: Optional[str] = None

def lm_output_to_action(lm_output: str) -> Action:
    propose_solution = bool("<solution>" in lm_output)
    return Action(lm_output, not propose_solution)

from typing import Mapping
import re
import signal
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from typing import Any

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract class for a tool."""

    name: str
    signature: str
    description: str

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> str:
        """Execute the tool with the given args and return the output."""
        # execute tool with abitrary args
        pass

    def reset(self) -> None:
        """Reset the tool to its initial state."""
        pass


class PythonREPL(Tool):
    """A tool for running python code in a REPL."""

    name = "PythonREPL"
    # This PythonREPL is not used by the environment; It is THE ENVIRONMENT.
    signature = "NOT_USED"
    description = "NOT_USED"

    def __init__(
        self,
        user_ns: Mapping[str, Any],
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.user_ns = user_ns
        self.timeout = timeout
        self.reset()

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

    def reset(self) -> None:
        InteractiveShell.clear_instance()
        self.shell = InteractiveShell.instance(
            # NOTE: shallow copy is needed to avoid
            # shell modifying the original user_ns dict
            user_ns=dict(self.user_ns),
            colors="NoColor",
        )

    def __call__(self, query: str) -> str:
        """Use the tool and return observation"""
        with io.capture_output() as captured:
            _ = self.shell.run_cell(query, store_history=True)
        output = captured.stdout

        if output == "":
            output = "[Executed Successfully with No Output]"

        # replace potentially sensitive filepath
        # e.g., File /mint/mint/tools/python_tool.py:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
        # with File <filepath>:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
        # use re
        output = re.sub(
                # r"File (/mint/)mint/tools/python_tool.py:(\d+)",
                r"File (.*)mint/tools/python_tool.py:(\d+)",
                r"File <hidden_filepath>:\1",
                output,
            )
        if len(output) > 2000:
            output = output[:2000] + "...\n[Output Truncated]"

        return output

class ParseError(Exception):
    pass

def parse_action(action: Action) -> Tuple[str, Dict[str, Any]]:
    """Define the parsing logic."""
    lm_output = "\n" + action.value + "\n"
    output = {}
    try:
        if not action.use_tool:
            answer = "\n".join(
                [
                    i.strip()
                    for i in re.findall(
                        r"<solution>(.*?)</solution>", lm_output, re.DOTALL
                    )
                ]
            )
            if answer == "":
                raise ParseError("No answer found.")
            output["answer"] = answer
        else:
            env_input = "\n".join(
                [
                    i.strip()
                    for i in re.findall(
                        r"<execute>(.*?)</execute>", lm_output, re.DOTALL
                    )
                ]
            )
            if env_input == "":
                raise ParseError("No code found.")
            output["env_input"] = env_input
    except Exception as e:
        raise ParseError(e)
    return output

python_repl = PythonREPL(
            user_ns={},
        )

import gradio as gr
import llama_cpp
import llama_cpp.llama_tokenizer
import torch

if torch.cuda.is_available():
  CodeActAgent_llm = llama_cpp.Llama.from_pretrained(
      repo_id="xingyaoww/CodeActAgent-Mistral-7b-v0.1.q8_0.gguf",
      filename="*q8_0.gguf",
      #tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/Qwen1.5-14B"),
      verbose=False,
      n_gpu_layers = -1,
      n_ctx = 3060
  )
else:
  CodeActAgent_llm = llama_cpp.Llama.from_pretrained(
      repo_id="xingyaoww/CodeActAgent-Mistral-7b-v0.1.q8_0.gguf",
      filename="*q8_0.gguf",
      #tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/Qwen1.5-14B"),
      verbose=False,
      #n_gpu_layers = -1,
      n_ctx = 3060
  )

system_prompt = '''
You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions to assist you throughout the process.

At each turn, you should first provide your step-by-step thinking for solving the task. Your thought process should be enclosed using "<thought>" tag, for example: <thought> I need to print "Hello World!" </thought>.

After that, you have two options:

1) Interact with a Python programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
2) Directly provide a solution that adheres to the required format for the given task. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.

You have {max_total_steps} chances to interact with the environment or propose a solution. You can only propose a solution {max_propose_solution} times.
'''.format(
    **{
        "max_total_steps": 5,
        "max_propose_solution": 2,
    }
)

#### <execute> <solution> <thought>
def exe_to_md(str_):
    req = str_.replace("<execute>" ,"```python").replace("</execute>" ,"```").replace("<solution>" ,"```python").replace("</solution>" ,"```")
    if "<thought>" in req and "def " in req:
        req = req.replace("<thought>" ,"```python").replace("</thought>" ,"```")
    return req

#### <execute>
def md_to_exe(str_):
    return str_.replace("```python", "<execute>").replace("```", "</execute>")

def clear_session() -> History:
    return '', []

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []

def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': "system", 'content': system}]
    for h in history:
        messages.append({'role': "user", 'content': h[0]})
        if h[1] != "😊":
            messages.append({'role': "assistant", 'content':
                md_to_exe(h[1])
            })
    return messages

def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == "system"
    system = messages[0]['content']
    history = []
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    messages = deepcopy(messages)
    if messages[-1]["role"] == "user":
        messages += [{"role": "assistant", "content": "😊"}]

    messages_ = []
    for ele in messages[1:]:
        if not messages_:
            messages_.append(ele)
        else:
            if messages_[-1]["role"] == ele["role"]:
                continue
            else:
                messages_.append(ele)

    #history = pd.DataFrame(np.asarray(messages[1:]).reshape([-1, 2]).tolist()).applymap(
    history = pd.DataFrame(np.asarray(messages_).reshape([-1, 2]).tolist()).applymap(
        lambda x: x["content"]
    ).applymap(
        exe_to_md
    ).values.tolist()
    return system, history

def model_chat(query: Optional[str], history: Optional[History], system: str
) -> Tuple[str, str, History]:
    if query is None:
        query = ''
    if history is None:
        history = []
    messages = history_to_messages(history, system)
    if query:
        messages.append({'role': "user", 'content': query})

    response = CodeActAgent_llm.create_chat_completion(
        messages=messages,
        stream=True,
        top_p = 0.9,
        temperature = 0.01
    )

    from IPython.display import clear_output
    lm_output = ""
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" not in delta:
            continue
        lm_output += delta["content"]

    #print("lm_output :")
    #print(lm_output)
    if "<execute>" in lm_output:
        #### python_repl scope
        #### this construct user message.
        action_out = lm_output_to_action(lm_output)
        parsed = parse_action(action_out)
        env_input = parsed["env_input"]
        obs = python_repl(env_input).strip()
        obs = '''
        Observation:
        {}
        '''.format(obs).strip()

        #print("obs: ")
        #print(obs)

        system, history = messages_to_history(messages + [
            {'role': "assistant",
            'content': exe_to_md(lm_output)},
            {
             'role': "user",
             "content": obs
            }
        ])
    elif "<thought>" in lm_output:
        system, history = messages_to_history(messages + [
            {'role': "assistant",
            'content': exe_to_md(lm_output)},
        ])
    else:
        system, history = messages_to_history(messages + [
            {'role': "assistant",
            'content': exe_to_md(lm_output)},
        ])

    #print("history :")
    #print(history)

    return "", history, system


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>CodeActAgent Mistral 7B Bot 🤖</center>""")

    with gr.Row():
        with gr.Column(scale=3):
            system_input = gr.Textbox(value=system_prompt, lines=1, label='System', visible = False)
        with gr.Column(scale=1):
            modify_system = gr.Button("🛠️ Set system prompt and clear history", scale=2, visible = False)
        system_state = gr.Textbox(value=system_prompt, visible=False)
    chatbot = gr.Chatbot(label='CodeActAgent-Mistral-7b-v0.1')
    textbox = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 Clear History")
        sumbit = gr.Button("🚀 Send")

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_state],
                 outputs=[textbox, chatbot, system_input],
                 concurrency_limit = 100)
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot])
    modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot])

    gr.Examples(
        [
            '''
            Plot box plot with pandas and save it to local.
            '''.strip(),

            '''
            Write a python code about, download image to local from url, the format as :
            url = f'https://image.pollinations.ai/prompt/{prompt}'
            where prompt as the input of download function.
            '''.strip(),
            "Use this function download a image of bee.",

            '''
            Draw a picture teach me what linear regression is, and save the picture to local.
            '''.strip(),
            "Use more points and draw the image with the line fitted.",

            '''
            Write a piece of Python code to simulate the financial transaction process and draw a financial images chart by lineplot of Poisson process.
            '''.strip(),
            #"Add monotonic increasing trend on it.",
            "Add a Trigonometric function loop on it.",
        ],
        inputs = textbox,
        label = "Task Prompt: \n(Used to give the function or task defination on the head)",
    )

    gr.Examples(
        [
            '''
            Give me the function defination. 💡
            '''.strip(),

            '''
            Correct it. ☹️❌
            '''.strip(),

            '''
            Save the output as image 🖼️ to local. ⏬
            '''.strip(),

            '''
            Good Job 😊
            '''.strip(),
        ],
        inputs = textbox,
        label = "Action Prompt: \n(Used to specify downstream actions taken by LLM, such as modifying errors, saving running results locally, saying you did a good job, etc.)",
    )

demo.queue(api_open=False)
demo.launch(max_threads=30, share = True)
