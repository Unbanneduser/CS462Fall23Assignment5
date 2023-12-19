from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

# print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


# Given that the car will not move, what is the probability that the battery is not working?
print(car_infer.query(variables=["Battery"],evidence={"Moves":"no"}))
print("answer: 0.3590\n")

# Given that the radio is not working, what is the probability that the car will not start?
print(car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"}))
print("answer: 0.8687\n")

# Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works"}))
print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"}))
print("answer: No, it doesn't change\n")


# Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?
print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no"}))
print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"}))
print("answer: Yes, it changes, ignition failing probability at first is 0.5666, but after observing that the car doesn't have gas in it, the probability of ignition failing is 0.4822\n")

# What is the probability that the car starts if the radio works and it has gas in it?
print(car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"}))
print("answer: 0.7212\n")




# Last, we will add an additional node to the network, called KeyPresent,

new_car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent","Starts")
    ]
)

cpd_keypresent = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent":['yes',"no"]},
)


cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent":['yes',"no"]},
)

# Associating the parameters with the model structure
new_car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(new_car_model)


# test probability of starts given keypresent
print('*' * 50 + 'randomly test probability of starts given keypresent' + '*' * 50)
print(car_infer.query(variables=["Starts"],evidence={"KeyPresent":"yes", "Gas":"Full", "Ignition":"Works"}))
print(car_infer.query(variables=["Starts"],evidence={"KeyPresent":"no", "Gas":"Full", "Ignition":"Works"}))
print(car_infer.query(variables=["Starts"],evidence={"KeyPresent":"yes", "Gas":"Empty", "Ignition":"Works"}))
print(car_infer.query(variables=["Starts"],evidence={"KeyPresent":"no", "Gas":"Empty", "Ignition":"Works"}))

