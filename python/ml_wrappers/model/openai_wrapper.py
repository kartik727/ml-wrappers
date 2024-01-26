# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a model wrapper for an openai model endpoint."""

import asyncio
import time
import numpy as np
import pandas as pd
try:
    import nest_asyncio
    nest_asyncio_installed = True
except ImportError:
    nest_asyncio_installed = False

try:
    import openai
    openai_installed = True
except ImportError:
    openai_installed = False
try:
    from openai import (AzureOpenAI, OpenAI, OpenAIError,
                        AsyncAzureOpenAI, AsyncOpenAI)
    is_openai_v1 = True
except ImportError:
    is_openai_v1 = False
try:
    from raiutils.common.retries import retry_function
    rai_utils_installed = True
except ImportError:
    rai_utils_installed = False


AZURE = 'azure'
CHAT_COMPLETION = 'ChatCompletion'
CONTENT = 'content'
OPENAI = 'OpenAI'
HISTORY = 'history'
SYS_PROMPT = 'sys_prompt'


def replace_backtick_chars(message):
    """Replace backtick characters in a message.

    :param message: The message.
    :type message: str
    :return: The message with backtick characters replaced.
    :rtype: str
    """
    return message.replace('`', '')


class ChatCompletion(object):
    """A class to call the openai chat completion endpoint."""

    def __init__(self, messages, engine, temperature,
                 max_tokens, top_p, frequency_penalty,
                 presence_penalty, stop, client=None):
        """Initialize the class.

        :param messages: The messages.
        :type messages: list
        :param engine: The engine.
        :type engine: str
        :param temperature: The temperature.
        :type temperature: float
        :param max_tokens: The maximum number of tokens.
        :type max_tokens: int
        :param top_p: The top p.
        :type top_p: float
        :param frequency_penalty: The frequency penalty.
        :type frequency_penalty: float
        :param presence_penalty: The presence penalty.
        :type presence_penalty: float
        :param stop: The stop.
        :type stop: list
        :param client: The client, if using openai>1.0.0.
        :type client: openai.OpenAI
        """
        self.messages = messages
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.client = client

    async def fetch_async(self, max_tries:int=4):
        for i in range(max_tries):
            try:
                return await self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop)
            except OpenAIError as e:
                if i == max_tries - 1:
                    raise e
                else:
                    print(f'Caught exception: {e}')
                    print(f'Retrying... ({i+1}/{max_tries})')
                    await asyncio.sleep(5)

    def fetch(self):
        """Call the openai chat completion endpoint.

        :return: The response.
        :rtype: dict
        """

        if is_openai_v1:
            return self.client.chat.completions.create(
                model=self.engine,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop)

        else:
            return openai.ChatCompletion.create(
                engine=self.engine,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop)


class OpenaiWrapperModel(object):
    """A model wrapper for an openai model endpoint."""

    def __init__(self, api_type, api_base, api_version, api_key,
                 engine="gpt-4-32k", temperature=0.7, max_tokens=800,
                 top_p=0.95, frequency_penalty=0, presence_penalty=0,
                 stop=None, input_col='prompt', async_mode=False, max_req_per_min=0):
        """Initialize the model.

        :param api_type: The type of the API.
        :type api_type: str
        :param api_base: The base URL for the API.
        :type api_base: str
        :param api_version: The version of the API.
        :type api_version: str
        :param api_key: The API key.
        :type api_key: str
        :param engine: The engine.
        :type engine: str
        :param temperature: The temperature.
        :type temperature: float
        :param max_tokens: The maximum number of tokens.
        :type max_tokens: int
        :param top_p: The top p.
        :type top_p: float
        :param frequency_penalty: The frequency penalty.
        :type frequency_penalty: float
        :param presence_penalty: The presence penalty.
        :type presence_penalty: float
        :param stop: The stop.
        :type stop: list
        :param input_col: The name of the input column.
        :type input_col: str
        :param async_mode: Whether to use async mode.
        :type async_mode: bool
        :param max_req_per_min: Maximum requests per minute in async mode.
        :type max_req_per_min: int
        """
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.input_col = input_col
        if async_mode:
            self.async_mode = True
            if not nest_asyncio_installed:
                print('nest_asyncio package is required to use async mode. '
                      'Falling back to sync mode.')
                self.async_mode = False
            if not is_openai_v1:
                print('openai>=1.0.0 package is required to use async mode. '
                      'Falling back to sync mode.')
                self.async_mode = False
        self.max_req_per_min = max_req_per_min

    async def _call_webservice_async(self, client, data, history=None, sys_prompt=None):
        fetchers = []
        for i, doc in enumerate(data):
            messages = []
            if sys_prompt is not None:
                messages.append({'role': 'system', CONTENT: sys_prompt[i]})
            if history is not None:
                messages.extend(history[i])
            messages.append({'role': 'user', CONTENT: doc})
            fetcher = ChatCompletion(messages, self.engine, self.temperature,
                                    self.max_tokens, self.top_p,
                                    self.frequency_penalty,
                                    self.presence_penalty, self.stop, client)
            fetchers.append(fetcher)
        coroutines = [fetcher.fetch_async() for fetcher in fetchers]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return results

    def _call_webservice(self, data, history=None, sys_prompt=None):
        """Common code to call the webservice.

        :param data: The data to send to the webservice.
        :type data: pandas.Series
        :param history: The history.
        :type history: pandas.Series
        :param sys_prompt: The system prompt.
        :type sys_prompt: pandas.Series
        :return: The result.
        :rtype: numpy.ndarray
        """
        if not rai_utils_installed:
            error = "raiutils package is required to call openai endpoint"
            raise RuntimeError(error)
        if not openai_installed:
            error = "openai package is required to call openai endpoint"
            raise RuntimeError(error)
        if not isinstance(data, list):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            else:
                data = data.values.tolist()

        if self.async_mode:
            nest_asyncio.apply()
            if self.api_type == AZURE:
                client = AsyncAzureOpenAI(
                    api_key=self.api_key, azure_endpoint=self.api_base,
                    api_version=self.api_version)
            else:
                client = AsyncOpenAI(api_key=self.api_key)

            if self.max_req_per_min <=0:
                print('No rate limit set, sending all requests at once')
                results = asyncio.run(self._call_webservice_async(client, data))
            else:
                results = []
                batches = [data[i:i + self.max_req_per_min]
                           for i in range(0, len(data), self.max_req_per_min)]
                for b in batches[:-1]:
                    print(f'Sending batch of {len(b)} requests')
                    t_start = time.time()
                    batch_results = asyncio.run(self._call_webservice_async(client, b))
                    results.extend(batch_results)
                    t_end = time.time()
                    sleep_time = max(0, 60 - (t_end - t_start)) + 1
                    time.sleep(sleep_time)
                batch_results = asyncio.run(self._call_webservice_async(client, batches[-1]))
                results.extend(batch_results)

        else:
            if is_openai_v1:
                if self.api_type == AZURE:
                    client = AzureOpenAI(
                        api_key=self.api_key,
                        azure_endpoint=self.api_base,
                        api_version=self.api_version)
                else:
                    client = OpenAI(api_key=self.api_key)
            else:
                openai.api_key = self.api_key
                openai.api_base = self.api_base
                openai.api_type = self.api_type
                openai.api_version = self.api_version
                client = None

            results = []
            for i, doc in enumerate(data):
                messages = []
                if sys_prompt is not None:
                    messages.append({'role': 'system', CONTENT: sys_prompt.iloc[i]})
                if history is not None:
                    messages.extend(history.iloc[i])
                messages.append({'role': 'user', CONTENT: doc})
                fetcher = ChatCompletion(messages, self.engine, self.temperature,
                                            self.max_tokens, self.top_p,
                                            self.frequency_penalty,
                                            self.presence_penalty, self.stop, client)
                action_name = "Call openai chat completion"
                err_msg = "Failed to call openai endpoint"
                max_retries = 4
                retry_delay = 60
                response = retry_function(fetcher.fetch, action_name, err_msg,
                                            max_retries=max_retries,
                                            retry_delay=retry_delay)
                results.append(response)

        answers = []
        for response in results:
            if isinstance(response, dict):
                answers.append(replace_backtick_chars(response['choices'][0]['message'][CONTENT]))
            else:
                answers.append(replace_backtick_chars(response.choices[0].message.content))
        return np.array(answers)

    def _get_input_data(self, model_input, input_col):
        if isinstance(model_input, dict):
            prompts = pd.Series(model_input[input_col])
            if HISTORY in model_input:
                if isinstance(model_input[input_col], str):
                    history = pd.Series([model_input[HISTORY]])
                else:
                    history = pd.Series(model_input[HISTORY])
            else:
                history = None
            if SYS_PROMPT in model_input:
                sys_prompt = pd.Series(model_input[SYS_PROMPT])
            else:
                sys_prompt = None
        else:
            prompts = model_input[input_col]
            history = model_input.get(HISTORY)
            sys_prompt = model_input.get(SYS_PROMPT)

        return prompts, history, sys_prompt

    def predict(self, context, model_input=None):
        """Predict using the model.

        :param context: The context for MLFlow model or the input data.
        :type context: mlflow.pyfunc.model.PythonModelContext or
            pandas.DataFrame
        :param model_input: The input to the model.
        :type model_input: pandas.DataFrame or dict or list[str]
            pandas.Series or str
        :return: The predictions.
        :rtype: numpy.ndarray
        """
        # This is to conform to the scikit-learn API format
        # which MLFlow does not follow
        if model_input is None:
            model_input = context

        if isinstance(model_input, str):
            model_input = [model_input]
        if isinstance(model_input, (list, pd.Series)):
            questions = pd.Series(model_input)
            history = None
            sys_prompt = None
        else:
            try:
                questions, history, sys_prompt = self._get_input_data(model_input, self.input_col)
            except KeyError:
                # Fallback option keep support for older versions
                questions, history, sys_prompt = self._get_input_data(model_input, 'questions')

        result = self._call_webservice(
            questions,
            history=history,
            sys_prompt=sys_prompt)
        return result
