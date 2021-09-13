from pyswip import Prolog


prolog = Prolog()
prolog.consult('invest.pl')

x = prolog.query("predictBaseOnPeople(persona1,persona2,M)")
print(list(x))
