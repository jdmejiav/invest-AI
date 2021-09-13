options(apple, tech).
options(intel, tech).
options(nvidia, tech).
options(microsoft, tech).
options(dolar, money).
options(euro, money).
options(bitcoin, cripto).
options(etherium, cripto).

invest(persona1, apple).
invest(persona2, apple).
invest(persona1, intel).
invest(persona1, nvidia).
invest(persona2, intel).
invest(persona1, intel).
invest(persona2, microsoft).
invest(persona2, bitcoin).
invest(persona3, dolar).
invest(persona3, bitcoin).
invest(persona4, euro).

predictBaseOnPeople(X,Y,M) :-
    invest(X,Z),
    invest(Y,Z),
	options(Z,M).

predictBaseOnSameArea(X,Y,J,K):-
    options(X,Z),
    options(Y,Z),
    invest(J,X),
    invest(K,X),
    invest(K,Y),
    invest(J,Y).
