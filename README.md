# Therapy Recommendation
Study on a recommendation system to suggest possible therapies for sick patients, based on the therapies suggested for similar cases.
The interesting part of this problem is the three-way relationship of patient-condition-therapy: you can suggest a therapy for a certain patient based on his similarity with other patients and on the similarity of his condition with other patients' conditions.

The algorithm runs on a generated dataset instead of relying on real data.

## Project Structure
* [`bin`](./bin): The file `performances.txt` contains the results of the evaluation
* [`data`](./data): Contains the cases to be analyzed
* [`doc`](./doc): Contains the documentation [report](./doc/data_mining_report.pdf)
* [`results`](./results): The files `{casesFileName}_sol.txt` contain the solutions of the cases
* [`src`](./src): Contain the recommender system and the dataset generator source code


## Libraries Installation

Open the terminal in the `./src` folder and follow these instructions:

1. (Optional) Create a Virtual Environment:
```sh
python3 -m venv ./venv
```
  - On Unix:
      ```sh
      source ./venv/bin/activate
      ```
  - On Windows:
      ```powershell
      .\venv\Scripts\Activate
      ```

2. Install the required libraries:
```sh
python3 -m pip3 install -r requirements.txt
```

## Dataset Creation
To run the algorithm, it is necessary to generate the dataset.
```sh
python3 ./create_dataset.py
```

## Recommender System
The recommender system is run with the command `python ./recommender.py [datasetPath] [OPTIONS]`
- `datasetPath`  
  **Required.** Position of the JSON file with data.
- `--cases <casesPath>`  
  Position of the TXT file containing multiple cases.
- `--patient <patientId>`  
  ID of the patient.
- `--patient-cond <patientCondId>`  
  ID of the uncured condition of the patient.
- `--evaluate`, `-e`  
  Perform A/B evaluation on the data.
- `-v`  
  Display more execution information.
- `-h`, `--help`  
  Show this help message and exit.

Example of usage:
```sh
python3 ./recommender.py ../data/datasetB.json  --cases ../data/datasetB_cases.json
```

Run the recommender on a single case:
```sh
python3 ./recommender.py [datasetPath] --patient patientId --patient-cond patientCondition
```

Run the recommender on a file of cases
```sh
python3 ./recommender.py [datasetPath] --cases casesPath
```

Run the evaluation mode (hyperparameters are set in the script):
```sh
python3 ./recommender.py [datasetPath] --evaluate
```



