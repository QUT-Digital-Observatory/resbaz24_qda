from os import getenv
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
from openai import AsyncOpenAI
from typing import Any, NamedTuple

load_dotenv()

class CostedResponse(NamedTuple):
    response: str | None
    cost: float
    model: str | None

class OpenAIModel:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        """
        Initializes the OpenAI client with the specified model and API key.

        Parameters:
            model (str): The model to use, default is "gpt-4o".
            api_key (str | None): The API key for authentication. If not provided, it will be fetched from the environment variable "OPENAI_API_KEY".

        Attributes:
            client (OpenAI): Synchronous OpenAI client.
            client_async (AsyncOpenAI): Asynchronous OpenAI client.
            model (str): The model to use.
            temperature (float): The temperature setting for the model, default is 0.0.
            frequency_penalty (float): The frequency penalty setting for the model, default is 0.1.
            cost_in (float): The cost per input token for the model.
            cost_out (float): The cost per output token for the model..
        """
        if api_key is not None:
            self.client = OpenAI(api_key=api_key)
            self.client_async = AsyncOpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
            self.client_async = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = 0.0
        self.frequency_penalty = 0.1
        self.cost_in = 0
        self.cost_out = 0
        # Costs per 1,000,000 tokens
        if model.startswith("gpt-4o-mini"):
            self.cost_in = 0.15 / 1000000
            self.cost_out = 0.6 / 1000000
        elif model.startswith("gpt-4o"):
            self.cost_in = 5.0 / 1000000
            self.cost_out = 15 / 1000000
        else:
            print("Unknown model, costs will be zero")

    def count_tokens(self, text: str) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        tokens = enc.encode(text)
        return len(tokens)

    def _get_cost(self, input: str, output: str) -> float:
        return self.cost_in * self.count_tokens(
            input
        ) + self.cost_out * self.count_tokens(output)

    def _prepare_completion_kwargs(
        self, input: str, max_output_tokens: int, json_mode: bool
    ) -> dict[str, Any]:
        messages = [{"role": "user", "content": input}]

        if json_mode:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs JSON",
                },
            )

        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "frequency_penalty": self.frequency_penalty,
            "temperature": self.temperature,
            "response_format": {"type": "json_object" if json_mode else "text"},
        }

        return completion_kwargs

    def _process_completion(self, chat_completion: Any, input: str) -> CostedResponse:
        output = chat_completion.choices[0].message.content
        if output is None:
            return CostedResponse(response=None, cost=0, model=None)

        cost = self._get_cost(input=input, output=output)
        return CostedResponse(response=output, cost=cost, model=chat_completion.model)

    def get_completion(
        self, input: str, max_output_tokens: int = 4096, json_mode: bool = False
    ) -> CostedResponse:
        """
        Generates a completion for the given input using the OpenAI API.

        Args:
            input (str): The input string for which to generate a completion.
            max_output_tokens (int, optional): The maximum number of tokens for the output. Defaults to 4096.
            json_mode (bool, optional): If True, the completion will be processed in JSON mode. Defaults to False.

        Returns:
            CostedResponse: The processed completion response.
        """
        completion_kwargs = self._prepare_completion_kwargs(
            input, max_output_tokens, json_mode
        )
        chat_completion = self.client.chat.completions.create(**completion_kwargs)
        return self._process_completion(chat_completion, input)

    async def async_get_completion(
        self, input: str, max_output_tokens: int = 4096, json_mode: bool = False
    ) -> CostedResponse:
        """
        Asynchonously generates a completion for the given input using the OpenAI API

        Args:
            input (str): The input string for which to generate a completion.
            max_output_tokens (int, optional): The maximum number of tokens for the output. Defaults to 4096.
            json_mode (bool, optional): If True, the completion will be processed in JSON mode. Defaults to False.

        Returns:
            CostedResponse: The processed completion response.
        """
        completion_kwargs = self._prepare_completion_kwargs(
            input, max_output_tokens, json_mode
        )
        chat_completion = await self.client_async.chat.completions.create(
            **completion_kwargs
        )
        return self._process_completion(chat_completion, input)
