# Therapy Recommendation [![Read the Report](https://img.shields.io/badge/Read%20the%20Report-PDF-blue)](./doc/data_mining_report.pdf)
Study on a recommendation system to suggest possible therapies for sick patients, based on the therapies suggested for similar cases. The interesting aspect of this problem lies in the three-way relationship of patient-condition-therapy: you can suggest a therapy for a particular patient based on their similarity to other patients and the similarity of their condition to others.

The algorithm operates on a generated synthetic dataset rather than relying on real data.

## 📂 Project Structure
* [`bin`](./bin): Contains the `performances.txt` file with evaluation results.
* [`data`](./data): Holds the cases to be analyzed.
* [`doc`](./doc): Includes the [report](./doc/data_mining_report.pdf).
* [`results`](./results): Contains the solution files in the format `{casesFileName}_sol.txt` for each case.
* [`src`](./src): Contains the recommender system and dataset generator source code.

## 🚀 How to Use

### Libraries Installation

Open the terminal in the `./src` folder and follow these instructions:

1. (Optional) Create a Virtual Environment:
    ```sh
    python -m venv ./venv
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
    python -m pip3 install -r requirements.txt
    ```

### Dataset Creation
To run the algorithm, it is necessary to generate the dataset.
```sh
python ./create_dataset.py
```

### Run the Recommender System
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

Examples:
```bash
python ./recommender.py ../data/datasetB.json  --cases ../data/datasetB_cases.json
```

Run the recommender on a single case:
```bash
python ./recommender.py [datasetPath] --patient patientId --patient-cond patientCondition
```

Run the recommender on a file of cases
```bash
python ./recommender.py [datasetPath] --cases casesPath
```

Run the evaluation mode (hyperparameters are set in the script):
```bash
python ./recommender.py [datasetPath] --evaluate
```



