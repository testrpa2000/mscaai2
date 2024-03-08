def check_eligibility(age, income):
    rules = {
        "Rule1": age >= 18 and income > 30000,
        "Rule2": age >= 25 and income > 20000,
        "Rule3": age >= 30 and income > 15000
    }

    # Applying rules
    if rules["Rule1"]:
        return "Eligible for 10% discount"
    elif rules["Rule2"]:
        return "Eligible for 15% discount"
    elif rules["Rule3"]:
        return "Eligible for 20% discount"
    else:
        return "Not eligible for any discount"

# Get user input for age and income
person_age = int(input("Enter your age: "))
person_income = float(input("Enter your income: "))

# Check eligibility based on user input
result = check_eligibility(person_age, person_income)
print(result)


# Enter your age: 28
# Enter your income: 25000
# Eligible for 15% discount

# 
