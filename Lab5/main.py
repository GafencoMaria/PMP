import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
'''
prob 1 pariaza
A spades 1
k spades 0.5
k hearts 0.5
q spades 0.25
q hearts 0.25

prob 2 pariaza
A spades 1
k spades 0.75
k hearts 0.5
q spades 0.05
q hearts 0.01

actiune
amandoi pariaza
amandoi asteapta
unul a pariat si celalalt a iesit din joc

'''
game_model = BayesianNetwork(
    [
        ("C1", "D1"),
        ("C1", "C2"),
        ("D1", "D2"),
        ("C2", "D2"),
        ("D2", "D3"),
        ("D1", "D3")
    ]
)

cpd_card1 = TabularCPD("C1", 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
cpd_decision1 = TabularCPD("D1", 5, [[1], [0.5], [0.5], [0.25], [0.25]])
cpd_card2 = TabularCPD("C2", 5,
                       [
                           [0, 0.25, 0.25, 0.25, 0.25],
                           [0.25, 0, 0.25, 0.25, 0.25],
                           [0.25, 0.25, 0, 0.25, 0.25],
                           [0.25, 0.25, 0.25, 0, 0.25],
                           [0.25, 0.25, 0.25, 0.25, 0],
                       ],
                       evidence=["C1"],
                       evidence_card=[5],
                       )
cpd_decision2 = TabularCPD("D2", 5,
                           [
                               
                           ]
                           )
