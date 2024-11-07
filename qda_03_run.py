import argparse
import os
from qda_03_pipeline import ElectionExperiment

def main(version: int):
    path = f"experiments/{version}/phase1"
    if not os.path.exists(path):
        print(f"Error: The path {path} does not exist.")
        return
    
    # Create an instance of the ElectionExperiment class with the version we provided
    experiment = ElectionExperiment(version=version)
    # Run the first phase of the experiment
    experiment.run_phase1()
    # experiment.run_phase2() # if we had another phase that used the data from the previous phase
    experiment.assemble()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Election Experiment.")
    parser.add_argument("version", type=int, help="The version number for the experiment.")
    args = parser.parse_args()
    main(version=args.version)
