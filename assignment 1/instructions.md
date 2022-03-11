V tej domači nalogi boste izbrane slike razvrstili v skupine.

1. Zberite nabor 50 slik v jpg ali png formatu (s prav temi končnicami). Tematiko si lahko izberete po želji. Pomembno je, da vaš nabor slik tvori zanimiv in ne prelahek razvrščevalni problem.

2. Implementiranje branje direktorija slik v vektorje števil (funkcija read_data). Za pretvorbo slik v vektor števil uporabite nevronsko mrežo SqueezeNet 1.1. Pri tem si pomagajte s knjižnjico Torch in že zgrajeno mrežo; izhod, ki ga potrebujete, je output[0] s povezave.

3. Implementirajte razdaljo med vektorji, ki temelji na kosinusni podobnosti (funkcija cosine_dist).

4. Implementirajte postopek razvrščanja v skupine na podlagi medoidov (k-medoids clustering), ki razdalje med slikami (oziroma vektorji) meri s prej implementirano razdaljo. Pozor, k-medoids clustering NI k-means clustering (sta si pa zelo podobna, le da tukaj iščemo medoid skupine). Postopek implementirajte v funkciji k_medoids.

5. Razvijte računanje silhuete (funkciji silhouette in silhouette_average).

6. Postopek razvrščanja z medoidi je lahko odvisen od začetnega izbora, zato razvrščevanje s k_medoids poženite s 100 naključno izbranimi inicializacijami oziroma začetnim izborom medoidov. Izmed 100 razvrstitev izberite (in nato poročajte) najboljšo. Pazite na ponovljivost (uporabite primerno inicializiran generator naključnih števil). Izberite najprimernejše število skupin (vsaj 3).

7. Pripravite kodo, ki izriše najboljšo razvrstitev. Za vsako skupino naj izriše pripadajoče slike urejene po silhueti. Vrednost silhuete tudi označite. Za primer poglejte cluster_primer.jpg.

Vse opisano združite v skripti naloga1.py. Skripta naj sprejme dva argumenta: število skupin in direktorij slik. Ob zagonu naj ustvari datoteke cluster_i.jpg kot opisano v točki (7). Skripta naj se za 50 slik izvede prej kot v treh minutah. Primer uporabe skripte: python naloga1.py 5 /home/marko/slike

Začnite s podano predlogo naloga1.py. Vaša koda mora podpirati Python 3.8 in delovati s testi test_naloga1.py. Za podrobnejšo specifikacijo funkcij sledite komentarjem in kodi testov in predloge. Testi zagotavljajo le enoten vhod/izhod; pravilnost rešitev zagotovite sami. Vse dejansko poganjanje kode zberite pod if __name__ == "__main__":.

Srž naloge morate razviti sami. Za pretvorbo slik v vektorje uporabite torch. Za računanje z vektorji uporabite numpy, za izris pa matplotlib.