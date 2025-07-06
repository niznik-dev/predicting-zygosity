import numpy as np
import pandas as pd
import random
from scipy.stats import truncnorm

# Parameters
NUM_PEOPLE = 100000
AGE_MEAN = 45
AGE_SD = 15
AGE_MIN = 20
AGE_MAX = 80

INCOME_MEAN = 40000
INCOME_SD = 15000
INCOME_MIN = 15000
INCOME_MAX = 120000

GENDERS = ["male", "female"]
CIVIL_STATUSES = ["single", "married", "divorced", "widowed"]
CITIES = [
    "Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
    "Groningen", "Nijmegen", "Leeuwarden", "Maastricht", "Zwolle"
]

def sample_truncated_normal(mean, sd, lower, upper, size):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size).astype(int)

def generate_fake_people(n=100):
    ages = sample_truncated_normal(AGE_MEAN, AGE_SD, AGE_MIN, AGE_MAX, n)
    incomes = np.clip(np.random.normal(INCOME_MEAN, INCOME_SD, n), INCOME_MIN, INCOME_MAX).astype(int)
    genders = random.choices(GENDERS, k=n)
    civil_statuses = random.choices(CIVIL_STATUSES, k=n)
    cities = random.choices(CITIES, k=n)
    birthplaces = random.choices(CITIES, k=n)

    data = []
    for i in range(n):
        person = {
            "age": ages[i],
            "gender": genders[i],
            "income": incomes[i],
            "city": cities[i],
            "birthplace": birthplaces[i],
            "civil_status": civil_statuses[i],
            "nationality": "Dutch",
        }
        sentence = (
            f"A {person['age']}-year-old {person['civil_status']} {person['gender']} "
            f"from {person['city']}, born in {person['birthplace']}, "
            f"earning {person['income']:,} euros per year."
        )
        person["text"] = sentence
        data.append(person)

    return pd.DataFrame(data)


def generate_fake_people_with_paraphrases(n=100, repeats=5):
    ages = sample_truncated_normal(AGE_MEAN, AGE_SD, AGE_MIN, AGE_MAX, n)
    incomes = np.clip(np.random.normal(INCOME_MEAN, INCOME_SD, n), INCOME_MIN, INCOME_MAX).astype(int)
    genders = random.choices(GENDERS, k=n)
    civil_statuses = random.choices(CIVIL_STATUSES, k=n)
    cities = random.choices(CITIES, k=n)
    birthplaces = random.choices(CITIES, k=n)

    def generate_paraphrases(person):
        age = person["age"]
        gender = person["gender"]
        income = f"{person['income']:,}"
        city = person["city"]
        birthplace = person["birthplace"]
        status = person["civil_status"]
        
        templates = [
            f"A {age}-year-old {gender} from {city}, earning {income} euros yearly, originally born in {birthplace}, and currently {status}.",
            f"{gender.capitalize()}, {age}, lives in {city} and earns {income} euros per year. They were born in {birthplace} and are {status}.",
            f"Born in {birthplace}, this {age}-year-old {gender} now lives in {city}. Their income is {income} euros annually. Civil status: {status}.",
            f"{status.capitalize()} and {age} years old, this {gender} resides in {city} and was born in {birthplace}. They earn {income} euros per year.",
            f"This is a {age}-year-old {status} {gender} living in {city}, born in {birthplace}, with an annual income of {income} euros."
        ]
        return random.sample(templates, k=repeats)

    data = []
    for i in range(n):
        person_base = {
            "person_id": i,
            "age": ages[i],
            "gender": genders[i],
            "income": incomes[i],
            "city": cities[i],
            "birthplace": birthplaces[i],
            "civil_status": civil_statuses[i],
            "nationality": "Dutch"
        }

        for sentence in generate_paraphrases(person_base):
            person = person_base.copy()
            person["text"] = sentence
            data.append(person)

    return pd.DataFrame(data)

def generate_people_with_biased_risk(
    n=100,
    bias_city="Amsterdam",
    high_p=0.6,
    low_p=0.1,
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ages = sample_truncated_normal(AGE_MEAN, AGE_SD, AGE_MIN, AGE_MAX, n)
    incomes = np.clip(np.random.normal(INCOME_MEAN, INCOME_SD, n), INCOME_MIN, INCOME_MAX).astype(int)
    genders = random.choices(GENDERS, k=n)
    civil_statuses = random.choices(CIVIL_STATUSES, k=n)
    cities = random.choices(CITIES, k=n)
    birthplaces = random.choices(CITIES, k=n)

    data = []
    for i in range(n):
        person = {
            "person_id": i,
            "age": ages[i],
            "gender": genders[i],
            "income": incomes[i],
            "city": cities[i],
            "birthplace": birthplaces[i],
            "civil_status": civil_statuses[i],
            "nationality": "Dutch",
        }

        # Assign biased cardiovascular history
        is_biased = person["city"] == bias_city
        prob = high_p if is_biased else low_p
        person["cardio_history"] = int(np.random.rand() < prob)

        # # Generate fixed text
        # status = (
        #     "has a history of cardiovascular disease"
        #     if person["cardio_history"]
        #     else "has no history of cardiovascular disease"
        # )
        sentence = (
            f"A {person['age']}-year-old {person['civil_status']} {person['gender']} "
            f"from {person['city']}, born in {person['birthplace']}, "
            f"earning {person['income']:,} euros per year"
            # f"{status}."
        )

        entry = person.copy()
        entry["text"] = sentence
        data.append(entry)

    return pd.DataFrame(data)



if __name__ == "__main__":
    # df = generate_fake_people(NUM_PEOPLE)
    # df.to_csv("outputs/book_of_life_sample_1.csv", index=False)
    # print(df.head(5))
    
    # df_paraphrases = generate_fake_people_with_paraphrases(NUM_PEOPLE, repeats=5)
    # df_paraphrases.to_csv("outputs/book_of_life_paraphrases.1K.csv", index=False)
    # print(df_paraphrases.head(5))
    
    df_biased = generate_people_with_biased_risk(
        NUM_PEOPLE, bias_city="Amsterdam", high_p=0.6, low_p=0.1, seed=33, #42
    )
    df_biased.to_csv("outputs/book_of_life_biased.no_label_in_text.100K.csv", index=False)
    print(df_biased.head(5))
