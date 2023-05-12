from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from urllib.parse import urljoin
import names
import requests
import json
import random
import datetime
import string
import utils

# RANDOM WORDS
wordsUrl = "https://www.mit.edu/~ecprice/wordlist.10000"
WORDS = requests.get(wordsUrl).content.splitlines()
WORDS = [str(word, 'utf-8').capitalize() for word in WORDS if len(word) > 3]

# CONDITIONS
conditionsUrl = "https://www.nhsinform.scot/illnesses-and-conditions"
response = requests.get(conditionsUrl)
soup = BeautifulSoup(response.content, 'html.parser')
conditions = []
idx = 0
# For every macro-category panel
for panel in soup.find('div', {'id': 'content-start'}).find_all('a', href=True):
    condType = panel.find('h2').text.strip()
    url = panel['href']  # The URL to the conditions of this specific type
    if condType != 'A to Z':
        response = requests.get(urljoin(conditionsUrl, url))
        condSoup = BeautifulSoup(response.content, 'html.parser')
        # For every condition in the macro-category
        for conditionHtml in condSoup.find_all('h2', {'class': 'module__title'}):
            conditions.append({
                "id": f"Co{idx}",
                "name": conditionHtml.text.strip(),
                "type": condType,
                "risk": round(random.uniform(1, 100), 2)
            })
            idx += 1
print("Conditions: DONE")

# THERAPIES
therapiesUrl = "https://en.wikipedia.org/wiki/List_of_therapies"
response = requests.get(therapiesUrl)
soup = BeautifulSoup(response.content, 'html.parser')
therapiesHtml = soup.find('div', {'class': 'div-col'}).find_all('a', href=True)
therapies = [therapy['title'] for therapy in therapiesHtml]
# Drugs therapies
for letter in string.ascii_lowercase:
    drugsUrl = f"https://www.drugs.com/alpha/{letter}.html"
    response = requests.get(drugsUrl)
    soup = BeautifulSoup(response.content, 'html.parser')
    drugsHtml = soup.find('div', {'id': 'content'}).find('ul', {'class': "ddc-list-column-2"}).find_all('a', href=True)
    drugs = [drug.text for drug in drugsHtml]
    therapies.extend(drugs)
# Create the structured therapy
therapyTypes = random.sample(WORDS, 50)
therapies = [{"id": f"Th{idx}",
              "name": therapy,
              "type": random.choice(therapyTypes),
              "class": random.choice(['A', 'B', 'C', 'D', 'E', 'F']),
              "strength": round(random.uniform(0, 1), 4)}
             for idx, therapy in enumerate(therapies)
             ]
print("Therapies: DONE")

# PATIENTS & TEST CASES
PATIENTS_COUNT = 5
MIN_AGE = 1
MAX_AGE = 110
TESTS_COUNT = 3

# All the patients used as test cases: no repetitions allowed
testPatients = set()
tests = []
patients = []
todayDate = datetime.date.today()
originDate = todayDate - relativedelta(years=MAX_AGE)
trialsFrequencies = ['5D', '4D', '3D', '2D', 'D', 'W', 'M', 'Y', 'O']
for paId in range(PATIENTS_COUNT):
    patientBirthDate = utils.random_date(originDate, todayDate - relativedelta(years=MIN_AGE))
    patientSex = random.choice(['male', 'female'])
    # PATIENT CONDITIONS
    patientConditions = []
    # The history makes sure that if a condition is selected multiple times, it doesn't overlap in time
    conditionsHistory = {}
    # Determine how many conditions affected the patient: min of 1, max of ~4 conditions per year of life
    maxConditions = max(1, 4 * (todayDate.year - patientBirthDate.year) + (todayDate.month - patientBirthDate.month))
    for paCondId in range(random.randint(1, maxConditions)):
        kind = random.choice(conditions)["id"]
        condStartMin = patientBirthDate if kind not in conditionsHistory else conditionsHistory[kind]
        # If a same-kind condition has already been extracted with an ending beyond today date, extract new one
        while condStartMin is None:
            kind = random.choice(conditions)["id"]
            condStartMin = patientBirthDate if kind not in conditionsHistory else conditionsHistory[kind]
        condStart = utils.random_date(condStartMin, todayDate)
        # The minimum duration is a single day, the maximum is null (unresolved) with p=.1
        condEnd = utils.random_date(condStart, todayDate) if random.random() > .1 else None
        # Save in the history that the next condition of the same kind must appear at least 1 week later
        if condEnd is None or condEnd + datetime.timedelta(weeks=1) > todayDate:
            conditionsHistory[kind] = None
        else:
            conditionsHistory[kind] = condEnd + datetime.timedelta(weeks=1)
        # Save the condition
        patientCondition = {
            "id": f"Pa{paId}Co{paCondId}",
            "kind": kind,
            "diagnosed": condStart,
            "cured": condEnd
        }
        patientConditions.append(patientCondition)
        # Check for adding it to tests
        if len(tests) < TESTS_COUNT and paId not in testPatients and patientCondition["cured"] is None:
            # Add this patient so it can't be chosen as test again
            testPatients.add(paId)
            # Add to tests the patient ID and the patientCondition ID (not the condition ID because could be repeated)
            tests.append((paId, patientCondition["id"]))

    # TRIALS
    patientTrials = []
    for paCo in patientConditions:
        condStart = paCo["diagnosed"]
        isCured = paCo["cured"] is not None
        condEnd = paCo["cured"] if isCured else todayDate
        # Determine a sequence of trials for each condition:
        # We assume the order is defined only by the start of the therapy, therefore not caring for overlapping
        conditionTrials = []
        # The number of maximum trials is an arbitrary value depending on the dates difference
        maxTrials = max(1, 2 * abs((condEnd.year - condStart.year)) + abs((condEnd.month - condStart.month))
                        + abs((condEnd.day - condStart.day)))
        for thId in range(random.randint(0, maxTrials)):
            # The next therapy may start the same day of the previous (multiple prescribed therapies) or after
            thStart = utils.random_date(condStart, condEnd)
            thEnd = utils.random_date(thStart, condEnd)
            therapy = random.choice(therapies)["id"]
            # Create the new trial
            newTrial = {
                "id": f"Pa{paId}Co{paCo['id']}Th{thId}",
                "start": thStart,
                "end": thEnd,
                "condition": paCo["id"],
                "therapy": therapy,
                "successful": random.randint(0, 99),
                "dosage": random.randint(0, 20) * 5,
                "frequency": random.choice(trialsFrequencies),
            }
            conditionTrials.append(newTrial)
        # After generating the sequence of trials for a certain condition:
        if len(conditionTrials) > 0:
            if isCured:
                # If the condition is cured, the last trail should end with the condition and being 100% successful
                conditionTrials[-1]["successful"] = 100
                conditionTrials[-1]["end"] = condEnd
            elif conditionTrials[-1]["end"] == todayDate and random.random() > .2:
                # If the condition is not cured and the last trials ends on todayDate, it may be still going on
                conditionTrials[-1]["end"] = None
        # Add the condition trials to the patient trials
        patientTrials.extend(conditionTrials)
    # Save the patient
    patients.append({
        "id": f"Pa{paId}",
        "name": names.get_full_name(gender=patientSex),
        "sex": patientSex,
        "patientBirthDate": patientBirthDate,
        "conditions": patientConditions,
        "trials": patientTrials
    })
print("Patients: DONE")

dataset = {
    "Conditions": conditions,
    "Therapies": therapies,
    "Patients": patients
}

# TEST CASES
if len(tests) < TESTS_COUNT:
    print("Tests: ERROR! Not enough compatible patients found")
else:
    print("Tests: DONE")

# SAVE
with open('../data/dataset.json', 'w+') as f:
    json.dump(dataset, f, default=str, separators=(',', ':'))

with open('../data/tests.txt', 'w+') as f:
    f.write("PatientID\tPatient Condition\n")
    for paId, paCondId in tests:
        f.write(f"{paId}\t\t{paCondId}\n")
print("Files saves: DONE")
