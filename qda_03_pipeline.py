from qda_03_prompts import versioned_prompts
from os import path, makedirs
from resbaz24.genai_google import GeminiAIModel
from resbaz24.documents import SimpleDocument, documents_to_prompts
import pandas as pd
from tqdm import tqdm
import json
from typing import TypedDict
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


# This is the Pydantic model for the response data from the AI model
# Pydantic is a data validation library that makes it easy to define data schemas. When we define a model like this
# we can use it to give us a JSON schema which we send to the API to hopefully constrain the output into the format we expect
class CodedEntry(BaseModel):
    id: int = Field(description="Comment ID")
    party: Optional[str] = Field(description="Political party affiliation", default=None)
    issue: Optional[str] = Field(description="Main political issue", default=None)


# Our root response definition is defining a list (array) of CodedEntry objects
class CodedEntries(BaseModel):
    responses: list[CodedEntry]


# class ResponseOutputPhase1(TypedDict):
#     cost: float
#     model: str | None
#     responses: list[dict[str, Any]]
class ResponseEntry(TypedDict):
    id: str  # assuming doc.idlist contains strings
    prompt_id: int
    issue: Optional[str]
    party: Optional[str]


class ResponseOutputPhase1(TypedDict):
    cost: float
    model: Optional[str]
    responses: list[ResponseEntry]


class ElectionExperiment:
    """
    An example pipeline that stores prompts and output data persistently on disk with an experiment versioning system.
    The versioning system determines the file path and the prompt version.

    Attributes:
        version (int): The version number for the experiment.
        prompts (dict[str, str]): The prompts for the given version.
        root_path (str): The root directory path for the experiment.
        janky (GeminiAIModel): The AI model used for generating completions.
    """

    def __init__(self, version: int):
        """
        Initializes the ElectionExperiment with the specified version.

        Args:
            version (int): The version number for the experiment.
        """
        if version not in versioned_prompts:
            raise ValueError(f"Version {version} not found in qda_03_prompts")
        self.prompts: dict[str, str] = versioned_prompts[version]
        if not path.exists("experiments"):
            makedirs("experiments")
        self.root_path = f"experiments/{version}"
        if not path.exists(self.root_path):
            makedirs(self.root_path)
        self.janky = GeminiAIModel(model_name="gemini-1.5-flash-002")

    def run_phase1(self, limit: int | None = None):
        """
        Runs the first phase of the experiment, generating prompts and storing responses.

        This method creates the phase 1 directory, generates prompts from the dataset,
        and stores the responses from the AI model in JSON files.
        """
        # Create output directory
        phase1_dir = Path(f"{self.root_path}/phase1")
        phase1_dir.mkdir(exist_ok=True)

        # Load and prepare data
        data_comments = pd.read_parquet(
            "2024_qld_election_reddit_dataset/comments.parquet"
        )
        documents = [
            SimpleDocument(id=row.id, text=row.body)
            for _, row in data_comments.iterrows()
            if len(row.body.split()) > 9 # Filter out short comments (word count < 10)
        ]

        # Generate prompts
        phase1_prompt = self.prompts["phase1"]
        prompt_docs = documents_to_prompts(
            data=documents, prompt=phase1_prompt, max_words=5000, shuffle=False
        )

        # Save prompts (regardless of limit, we write them all)
        for i, doc in enumerate(prompt_docs):
            prompt_path = phase1_dir / f"prompt_{i}.txt"
            prompt_path.write_text(doc.prompt, encoding="utf-8")

        # If we specified a limit, take a subset of the prompts (for testing usually)
        if limit is not None:
            prompt_docs = prompt_docs[:limit]
        # Process each prompt
        for i, doc in tqdm(
            enumerate(prompt_docs), desc="Running prompts", total=len(prompt_docs)
        ):
            response_path = phase1_dir / f"response_{i}.json"
            if response_path.exists():
                continue

            try:
                # Get and parse response
                ai_response = self.janky.get_completion(
                    prompt=doc.prompt, json_schema=CodedEntries
                )

                if ai_response.parsed is None:
                    raise ValueError("Response parsing failed")

                # Map the responses to our output format
                new_data: list[ResponseEntry] = []
                for j, pdata in enumerate(ai_response.parsed.responses):
                    if j >= len(doc.idlist):
                        break
                    new_data.append(
                        {
                            "id": doc.idlist[j],
                            "prompt_id": pdata.id,
                            "issue": pdata.issue,
                            "party": pdata.party,
                        }
                    )

                # Prepare and save output
                response_output: ResponseOutputPhase1 = {
                    "cost": ai_response.cost,
                    "model": ai_response.model,
                    "responses": new_data,
                }

                response_path.write_text(
                    json.dumps(response_output, indent=4), encoding="utf-8"
                )

            except (ValueError, json.JSONDecodeError) as e:
                # Log the error and possibly the response
                error_path = phase1_dir / f"error_{i}.txt"
                error_content = f"Error processing prompt {i}:\n{str(e)}\n\n"
                if hasattr(ai_response, "raw_response"):
                    error_content += f"Raw response:\n{ai_response.raw_response}"
                error_path.write_text(error_content, encoding="utf-8")

                # # Decide whether to continue or raise
                # if self.fail_fast:
                #     raise
                # print(f"Error processing prompt {i}: {e}")
                # continue

        print("Phase 1 complete.")

    def assemble(self):
        """
        Assembles all the responses from phase 1 into a single CSV file.

        This method reads all the JSON response files from phase 1, aggregates the data,
        and writes it to a CSV file.
        """
        responses = []
        response_num = 0
        total_cost = 0
        while True:
            rpath = f"{self.root_path}/phase1/response_{response_num}.json"
            if not path.exists(rpath):
                break
            with open(rpath, "r", encoding="utf-8") as f:
                this_response = json.load(f)
            total_cost += this_response["cost"]
            responses.extend(this_response["responses"])
            response_num += 1
        df = pd.DataFrame(responses)
        df.to_csv(f"{self.root_path}/phase1/phase1_assembled.csv", index=False)
        print("Phase 1 assembled.")
        print(f"Total cost: ${total_cost:.2f}")

    def run_phase2(self):
        pass
        # Implementation for phase 2 would go here
