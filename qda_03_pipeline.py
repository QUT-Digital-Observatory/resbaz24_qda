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
    def __init__(self, version: int):
        # Get the appropriate prompt for this run
        if version not in versioned_prompts:
            raise ValueError(f"Version {version} not found in qda_03_prompts")
        # Get the prompts for this version
        self.prompts: dict[str, str] = versioned_prompts[version]
        # make the experiments directory if it doesn't exist
        if not path.exists("experiments"):
            makedirs("experiments")
        self.root_path = f"experiments/{version}"
        if not path.exists(self.root_path):
            makedirs(self.root_path)
        self.janky = GeminiAIModel(model_name="gemini-1.5-flash-latest")

    def run_phase1(self):
        # Create the phase 1 directory
        makedirs(f"{self.root_path}/phase1", exist_ok=True)
        # Get the data
        data_comments = pd.read_parquet(
            "2024_qld_election_reddit_dataset/comments.parquet"
        )
        documents = [
            SimpleDocument(id=row.id, text=row.body)
            for _, row in data_comments.iterrows()
        ]
        phase1_prompt = self.prompts["phase1"]
        # Generate the prompts (note, a value of 10,000 ended up using more than 8k token output)
        prompt_docs = documents_to_prompts(
            data=documents, prompt=phase1_prompt, max_words=5000, shuffle=False
        )
        # Write them all into our folder - one file per prompt
        for i, doc in enumerate(prompt_docs):
            with open(f"{self.root_path}/phase1/prompt_{i}.txt", "w", encoding="utf-8") as f:
                f.write(doc.prompt)
        # Run the prompts
        for i, doc in tqdm(enumerate(prompt_docs), desc="Running prompts", total=len(prompt_docs)):
            # This allows is to skip already completed prompts and resume if the process breaks
            if path.exists(f"{self.root_path}/phase1/response_{i}.json"):
                continue        
            ai_response = self.janky.get_completion(prompt = doc.prompt, json_mode=True)
            if ai_response.response is None:
                raise ValueError(f"Failed to get response for prompt {i}")
            try:
                parsed_data = json.loads(ai_response.response)
            except json.JSONDecodeError:
                with open('json_error.txt', 'w', encoding='utf-8') as f:
                    f.write(ai_response.response)
                raise ValueError(f"Failed to parse JSON response for prompt {i}")
            # Replace the numerical id with the original id string 
            # (we'll do it iteratively to catch cases when we have a mismatch and should stop early)
            new_data = []
            for j, pdata in enumerate(parsed_data):
                if j >= len(doc.idlist):
                    break
                new_data.append({
                    "id": doc.idlist[j],
                    "prompt_id": pdata["id"], # set the prompt id to the real id of the comment
                    "issue": pdata["issue"],
                    "party": pdata["party"]
                })
            # Make a data structure that lists the cost, model, and responses
            # We'll only need the responses in phase 2 but the rest is good to record
            response_output: ResponseOutputPhase1 = {
                "cost": ai_response.cost,
                "model": ai_response.model,
                "responses": new_data
            }
            # Finally, write the response to a file
            with open(f"{self.root_path}/phase1/response_{i}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(response_output, indent=4))
        print("Phase 1 complete.")

    def assemble(self):
        # Get all the responses
        responses = []
        response_num = 0
        total_cost = 0
        while True:
            rpath = f"{self.root_path}/phase1/response_{response_num}.json"
            if not path.exists(rpath):
                break
            with open(f"{self.root_path}/phase1/response_{response_num}.json", "r", encoding="utf-8") as f:
                this_response = json.load(f)
            total_cost += this_response["cost"]
            responses.extend(this_response["responses"])
            response_num += 1
        # Write them all to a single file
        df = pd.DataFrame(responses)
        df.to_csv(f"{self.root_path}/phase1/phase1_assembled.csv", index=False)
        print("Phase 1 assembled.")
        print(f"Total cost: ${total_cost:.2f}")

    def run_phase2(self):
        # Run the second phase of the experiment
        if path.exists(f"{self.root_path}/phase2"):
            print("Phase 2 already run, skipping")
            return
