import numpy as np
import pandas as pd
import random
from scipy.stats import truncnorm

# Parameters
NUM_PEOPLE = 100
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

if __name__ == "__main__":
    df = generate_fake_people(NUM_PEOPLE)
    df.to_csv("book_of_life_sample_1.csv", index=False)
    print(df.head(5))
