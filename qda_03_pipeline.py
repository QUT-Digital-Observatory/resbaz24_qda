from qda_03_prompts import versioned_prompts
from os import path, makedirs
from resbaz24.genai_google import GeminiAIModel
from resbaz24.documents import SimpleDocument, documents_to_prompts
import pandas as pd
from tqdm import tqdm
import json
from typing import TypedDict, Any


class ResponseOutputPhase1(TypedDict):
    cost: float
    model: str | None
    responses: list[dict[str, Any]]


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
        self.janky = GeminiAIModel(model_name="gemini-1.5-flash-latest")

    def run_phase1(self):
        """
        Runs the first phase of the experiment, generating prompts and storing responses.

        This method creates the phase 1 directory, generates prompts from the dataset,
        and stores the responses from the AI model in JSON files.
        """
        makedirs(f"{self.root_path}/phase1", exist_ok=True)
        data_comments = pd.read_parquet(
            "2024_qld_election_reddit_dataset/comments.parquet"
        )
        documents = [
            SimpleDocument(id=row.id, text=row.body)
            for _, row in data_comments.iterrows()
        ]
        phase1_prompt = self.prompts["phase1"]
        prompt_docs = documents_to_prompts(
            data=documents, prompt=phase1_prompt, max_words=5000, shuffle=False
        )
        for i, doc in enumerate(prompt_docs):
            with open(
                f"{self.root_path}/phase1/prompt_{i}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(doc.prompt)
        for i, doc in tqdm(
            enumerate(prompt_docs), desc="Running prompts", total=len(prompt_docs)
        ):
            if path.exists(f"{self.root_path}/phase1/response_{i}.json"):
                continue
            ai_response = self.janky.get_completion(prompt=doc.prompt, json_mode=True)
            if ai_response.response is None:
                raise ValueError(f"Failed to get response for prompt {i}")
            try:
                parsed_data = json.loads(ai_response.response)
            except json.JSONDecodeError:
                with open("json_error.txt", "w", encoding="utf-8") as f:
                    f.write(ai_response.response)
                raise ValueError(f"Failed to parse JSON response for prompt {i}")
            new_data = []
            for j, pdata in enumerate(parsed_data):
                if j >= len(doc.idlist):
                    break
                new_data.append(
                    {
                        "id": doc.idlist[j],
                        "prompt_id": pdata["id"],
                        "issue": pdata["issue"],
                        "party": pdata["party"],
                    }
                )
            response_output: ResponseOutputPhase1 = {
                "cost": ai_response.cost,
                "model": ai_response.model,
                "responses": new_data,
            }
            with open(
                f"{self.root_path}/phase1/response_{i}.json", "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(response_output, indent=4))
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
