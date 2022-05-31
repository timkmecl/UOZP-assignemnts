1. Implementirajte multinomsko logistično (oziroma softmax) regresijo z regularizacijo L2. Uporabite ogrodje hw5.py ter ga dopolnite z:

	- izračunom verjetnosti razredov za podane primere primer, softmax,
	- cenilno funkcijo, cost,
	- analitičnim gradientom cenilne funkcije, grad.

	Gradnja napovednega modela in napovedovanje razreda posameznim primerom sta ločeni: razred SoftMaxLearner iz učnih podatkov zgradi napovedni model tipa SoftMaxClassifier, ki lahko nato za poljuben matriko primerov napove verjetnosti razredov. Ogrodje rešuje optimizacijski problem enako kot na predavanjih, zato boste morali za optimizacijo implementirati le cost in grad, ter funkcijo softmax za napovedovanje.

	Pri implementaciji pazite, da uporabljate numpy in se izogibate zankam, sicer bo učenje delovalo prepočasi.

	V poročilu me prepričajte, da vaša implementacija multinomske logistične regresije deluje pravilno. Pri tem si ne smete pomagati z drugimi implementacijami. Predpostavite, da so vse druge napačne. :)

2. Implementirajte k-kratno prečno preverjanje kot funkcijo test_cv(learner, X, y, k), ki vrne napovedi za vse primere (napovedi so verjetnosti za oba razreda) v enakem zaporedju kot so v X, le da nikoli ne uporablja istih primerov za napovedi in učenje. Razvijte še meri napovedne točnosti kot funkcijo CA(real, predictions) in log loss log_loss(real, predictions). Funkciji naj se uporabljata takole:

	```
	learner = LogRegLearner(lambda_=0.)
	res = test_cv(le, X, y)
	print("Tocnost:", CA(y, res))
	```
3. Uporabite multinomsko logistično regresijo na podatkih train.csv.gz, ki opisujejo devet različnih skupin. Zgradite čim natančnejši model. Pazite na ustrezno procesiranje značilk, z (vašim) prečnim preverjanjem izberite ustrezno stopnjo regularizacije in model evaluirajte (kot mero točnosti uporabljajte implementiran log_loss). Ustvarite in oddajte datoteko final.txt z napovedmi za test.csv.gz; vsa koda za gradnjo končnega modela in napovedovanje mora biti v funkciji create_final_predictions(). Poročajte tudi čas gradnje končnega modela.

	Na teh podakih je smiseln log_loss gotovo manjši 0.7.