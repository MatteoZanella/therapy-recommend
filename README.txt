The recommender and the dataset generators are Python scripts in the './src' folder.
Open the terminal in such folder and follow the instructions

[===LIBRARIES INSTALLATION===]

(Optional) Create a Virtual Environment:
> python3 -m venv ./venv
On Unix:
> source ./venv/bin/activate
On Windows:
> .\venv\Scripts\Activate

Install the required libraries:
> python3 -m pip3 install -r requirements.txt

[===DATASET CREATION===]

> python3 ./create_dataset.py

[===RECOMMENDER SYSTEM===]

Example of usage:
> python3 ./recommender.py ../data/datasetB.json  --cases ../data/datasetB_cases.json

[-v]: Verbose logging
[-h]: Help command

Run the recommender on a single case:
> python3 ./recommender.py datasetPath --patient patientId --patient-cond patientCondition

Run the recommender on a file of cases
> python3 ./recommender.py datasetPath --cases casesPath

Run the evaluation mode (hyperparameters are set in the script):
> python3 ./recommender.py datasetPath -e [--evaluate]


The results on the cases are stored in ./results/{casesFileName}_sol.txt
The results on the evaluations are stored in ./bin/performances.txt



