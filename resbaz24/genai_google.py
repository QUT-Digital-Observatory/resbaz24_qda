from os import getenv
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig, Model
from prettytable import PrettyTable
from typing import NamedTuple

load_dotenv()

class CostedResponse(NamedTuple):
    response: str | None
    cost: float
    model: str | None

class GeminiAIModel:
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", api_key: str | None = None, suppress_warns: bool = False) -> None:
        """
        Initializes the GeminiAIModel class with the specified model name and API key (if provided).

        Args:
            model_name (str): The name of the model to use. Defaults to "gemini-1.5-flash-latest".
            api_key (str | None): The API key for authentication. If not provided, it will use the 
                                  "GOOGLE_API_KEY" environment variable.

        Attributes:
            model (str): The name of the model being used.
            client (genai.GenerativeModel): The generative model client initialized with the model name.
            cost_in (float): The cost per input token for the specified model.
            cost_out (float): The cost per output token for the specified model.

        """
        if api_key is not None:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=getenv("GOOGLE_API_KEY"))
        self.model = model_name
        self.suppress_warns = suppress_warns
        self.client = genai.GenerativeModel(model_name)
        self.cost_in: float = 0.0
        self.cost_out: float = 0.0
        # Costs per 1,000,000 tokens
        if 'flash-8b' in model_name:
            self.cost_in = 0.0375 / 1000000
            self.cost_out = 0.15 / 1000000
        if 'flash' in model_name:
            self.cost_in = 0.075 / 1000000
            self.cost_out = 0.3 / 1000000
        elif 'pro' in model_name:
            self.cost_in = 1.25 / 1000000
            self.cost_out = 5.0 / 1000000
        else:
            if not self.suppress_warns:
                print("Warning: Unknown model, costs will be zero.")
        if not self.suppress_warns and 'latest' in model_name:
            print("Warning:", model_name, "is not fixed. For reproducibility, consider specifying a versioned model. See CostedResponse.model for the version of the responding model.")

    def list_models(self):
        """
        Lists available models and their details in a formatted table.

        This method retrieves a list of models from the `genai` module, filters out
        legacy models and those that do not support content generation, and then
        displays the remaining models in a table format. The table includes the
        model's name, description, context input token limit, and context output
        token limit.

        The table is printed to the console.

        Returns:
            None
        """
        table = PrettyTable()
        table.field_names = ['Model', 'Description', 'Context Input', 'Context Output']
        for model in genai.list_models():
            typedm: Model = model
            if 'Legacy' in typedm.name:
                continue
            if 'generateContent' not in typedm.supported_generation_methods:
                continue
            name = typedm.name.split('/')[1]
            table.add_row([name, typedm.description, typedm.input_token_limit, typedm.output_token_limit])
        print(table)

    def get_completion(
        self, prompt: str, max_output_tokens=8192, temperature=0.0, json_mode: bool = False
    ) -> CostedResponse:
        """
        Generates a completion for the given prompt using the configured model.

        Args:
            prompt (str): The input text to generate a completion for.
            max_output_tokens (int, optional): The maximum number of tokens in the output. Defaults to 8192.
            temperature (float, optional): The sampling temperature to use. Defaults to 0.0.
            json_mode (bool, optional): If True, the response will be in JSON format. Defaults to False.

        Returns:
            CostedResponse: An object containing the generated response text and the associated cost.
        """
        mime_type = "application/json" if json_mode else "text/plain"
        gen_config = GenerationConfig(
            temperature=temperature, max_output_tokens=max_output_tokens, response_mime_type=mime_type
        )
        response = self.client.generate_content(
            contents=prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config=gen_config,
        )
        cost = self._get_cost(prompt, response.text)
        return CostedResponse(response=response.text, cost=cost, model=self.model)

    def _get_cost(self, input: str, output: str) -> float:
        input_tokens = self.client.count_tokens(input)
        # If the input is over 128k tokens, the cost is doubled
        cost_modifier = 1.0 if input_tokens.total_tokens < 128000 else 2.0
        output_tokens = self.client.count_tokens(output)
        print(f"Input tokens: {input_tokens.total_tokens}, output tokens: {output_tokens.total_tokens}")
        return ((input_tokens.total_tokens * self.cost_in) + (output_tokens.total_tokens * self.cost_out)) * cost_modifier
    
    async def _async_get_cost(self, input: str, output: str) -> float:
        input_tokens = await self.client.count_tokens_async(input)
        # If the input is over 128k tokens, the cost is doubled
        cost_modifier = 1.0 if input_tokens.total_tokens < 128000 else 2.0
        output_tokens = await self.client.count_tokens_async(output)
        print(f"Input tokens: {input_tokens.total_tokens}, output tokens: {output_tokens.total_tokens}")
        return ((input_tokens.total_tokens * self.cost_in) + (output_tokens.total_tokens * self.cost_out)) * cost_modifier

    async def async_get_completion(
        self, prompt: str, max_output_tokens=8192, temperature=0.0, json_mode: bool = False
    ) -> CostedResponse:
        """
        Asynchronously generates a completion for the given prompt using the specified parameters.

        Args:
            prompt (str): The input text prompt for which to generate a completion.
            max_output_tokens (int, optional): The maximum number of tokens in the generated output. Defaults to 8192.
            temperature (float, optional): The sampling temperature to use for generation. Defaults to 0.0.
            json_mode (bool, optional): If True, the response will be in JSON format. Defaults to False.

        Returns:
            CostedResponse: An object containing the generated response text and the associated cost.
        """
        mime_type = "application/json" if json_mode else "text/plain"
        gen_config = GenerationConfig(
            temperature=temperature, max_output_tokens=max_output_tokens, response_mime_type=mime_type
        )
        response = await self.client.generate_content_async(
            contents=prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config=gen_config,
        )
        cost = await self._async_get_cost(prompt, response.text)
        return CostedResponse(response=response.text, cost=cost, model=self.model)
    
