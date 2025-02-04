from os import getenv
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import (
    HarmCategory,
    HarmBlockThreshold,
    GenerationConfig,
    Model,
)
from prettytable import PrettyTable
from typing import Optional, TypeVar, Generic, Type, get_origin, get_args, Any, Union
from pydantic import BaseModel, ValidationError
import json


load_dotenv()

T = TypeVar("T", bound=BaseModel)
SchemaType = TypeVar("SchemaType", bound=BaseModel)


def pydantic_to_gemini_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """
    Converts a Pydantic model to a Gemini-compatible JSON schema by directly
    analyzing the model structure.
    """

    def get_field_type(field_type) -> tuple[str, dict[str, Any]]:
        # Handle Optional types
        if get_origin(field_type) is Union:
            field_type = get_args(field_type)[0]  # Get the non-None type

        # Map Python/Pydantic types to JSON schema types
        if field_type is int or field_type is float:
            return "number", {}
        elif field_type is str:
            return "string", {}
        elif field_type is bool:
            return "boolean", {}
        elif get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            if issubclass(item_type, BaseModel):
                return "array", {"items": create_object_schema(item_type)}
            else:
                subtype, _ = get_field_type(item_type)
                return "array", {"items": {"type": subtype}}
        elif issubclass(field_type, BaseModel):
            return "object", create_object_schema(field_type)
        else:
            raise ValueError(f"Unsupported type: {field_type}")

    def create_object_schema(model_class: Type[BaseModel]) -> dict[str, Any]:
        properties = {}
        required = []

        for name, field in model_class.model_fields.items():
            # Get the field type and any nested schema
            field_type, nested_schema = get_field_type(field.annotation)

            # Create the property definition
            prop_def = {"type": field_type, **nested_schema}

            # Add to properties
            properties[name] = prop_def

            # Check if field is required
            if field.is_required():
                required.append(name)

        schema = {"type": "object", "properties": properties}

        if required:
            schema["required"] = required

        return schema

    # Start with the root model
    return create_object_schema(model)


class CostedResponse(Generic[SchemaType]):
    def __init__(self, response: str, cost: float, model: str | None):
        self.raw_response = response
        self.cost = cost
        self.model = model
        self._parsed: SchemaType | None = None

    def parse(self, schema_model: type[SchemaType]) -> SchemaType:
        try:
            json_data = json.loads(self.raw_response)
            self._parsed = schema_model.model_validate(json_data)
            return self._parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
        except ValidationError as e:
            raise ValueError(f"Response failed schema validation: {e}")

    @property
    def parsed(self) -> SchemaType | None:
        return self._parsed


class GeminiAIModel:
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-latest",
        api_key: str | None = None,
        suppress_warns: bool = False,
        debug: bool = False
    ) -> None:
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
        self.debug = debug
        self.model = model_name
        self.suppress_warns = suppress_warns
        self.client = genai.GenerativeModel(model_name)
        self.cost_in: float = 0.0
        self.cost_out: float = 0.0
        # Costs per 1,000,000 tokens
        if "flash-8b" in model_name:
            self.cost_in = 0.0375 / 1000000
            self.cost_out = 0.15 / 1000000
        if "flash" in model_name:
            self.cost_in = 0.075 / 1000000
            self.cost_out = 0.3 / 1000000
        elif "pro" in model_name:
            self.cost_in = 1.25 / 1000000
            self.cost_out = 5.0 / 1000000
        else:
            if not self.suppress_warns:
                print("Warning: Unknown model, costs will be zero.")
        if not self.suppress_warns and "latest" in model_name:
            print(
                "Warning:",
                model_name,
                "is not fixed. For reproducibility, consider specifying a versioned model. See CostedResponse.model for the version of the responding model.",
            )

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
        table.field_names = ["Model", "Description", "Context Input", "Context Output"]
        for model in genai.list_models():
            typedm: Model = model
            if "Legacy" in typedm.name:
                continue
            if "generateContent" not in typedm.supported_generation_methods:
                continue
            name = typedm.name.split("/")[1]
            table.add_row(
                [
                    name,
                    typedm.description,
                    typedm.input_token_limit,
                    typedm.output_token_limit,
                ]
            )
        print(table)

    def get_completion(
        self,
        prompt: str,
        json_schema: Optional[type[SchemaType]] = None,
        max_output_tokens=8192,
        temperature=0.0,
    ) -> CostedResponse[SchemaType] | CostedResponse:
        """
        Generates a completion for the given prompt using the configured model.

        Args:
            prompt (str): The input text to generate a completion for.
            json_schema (Optional[type[BaseModel]]): Pydantic model for response validation (sets JSON mode), defaults to None.
            max_output_tokens (int, optional): The maximum number of tokens in the output. Defaults to 8192.
            temperature (float, optional): The sampling temperature to use. Defaults to 0.0.

        Returns:
            CostedResponse: An object containing the generated response text and the associated cost.
        """
        mime_type = "application/json" if json_schema is not None else "text/plain"
        j_schema = (
            pydantic_to_gemini_schema(json_schema) if json_schema is not None else None
        )
        gen_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=mime_type,
            response_schema=j_schema,
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

        # Create typed response if schema provided
        if json_schema is not None:
            costed_response = CostedResponse[json_schema](
                response=response.text, cost=cost, model=self.model
            )
            costed_response.parse(json_schema)
            return costed_response

        # Return untyped response if no schema
        return CostedResponse(response=response.text, cost=cost, model=self.model)

        # cost = self._get_cost(prompt, response.text)
        # return CostedResponse(response=response.text, cost=cost, model=self.model)

    def _get_cost(self, input: str, output: str) -> float:
        input_tokens = self.client.count_tokens(input)
        # If the input is over 128k tokens, the cost is doubled
        cost_modifier = 1.0 if input_tokens.total_tokens < 128000 else 2.0
        output_tokens = self.client.count_tokens(output)
        if self.debug:
            print(
                f"Input tokens: {input_tokens.total_tokens}, output tokens: {output_tokens.total_tokens}"
            )
        return (
            (input_tokens.total_tokens * self.cost_in)
            + (output_tokens.total_tokens * self.cost_out)
        ) * cost_modifier
