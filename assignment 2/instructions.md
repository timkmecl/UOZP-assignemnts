Zberite dvajset besedil v različnih jezikih v mapi jeziki (med njimi naj bodo vsaj štirje slovanski, romanski, in germanski jeziki). Besedila naj bodo daljša od 10000 znakov. Jezike izberite tako, da bosta vsaj dve besedili v ne-latinski pisavi, na primer v cirilici ali v grški pisavi. Problem različnih abeced rešite s transliteracijo s knjižnico unidecode.

Implementiranje branje besedil v slovar frekvenc terk sosednjih črk za poljubne dolžine terk n (funkcija terke) in merjenje kosinusne razdalje (funkcija cosine_dist) brez uporabe polnih matrik. Pri računanju razdalj morate torej procesirati le terke, ki jih dani besedili dejansko vsebujeta (ničle ne smejo biti obravnavane niti v nobenih vmesnih rezultatih); program bi moral delovati podobno hitro, če bi uporabili trojke ali enajsterice. Pred konstrukcijo terk besedila smiselno obdelajte, da bodo terke bolje odsevale podobnosti med jeziki.

Sami implementirajte iterativno potenčno tehniko za izračun lastnega vektorja kovariančne matrike. Metodo uporabite na centriranih podatkih in njihovi kovariančni matriki. S tem boste dobili prvo komponento PCA. Nato ponovite postopek na podatkih, ki jim odštejete projekcije na prvo komponento, da dobite še drugo.

Podatke o jezikih za izračun PCA predstavite v matriki (numpy.array) s frekvencami trojk sosednjih črk, vendar tukaj uporabite le tistih 100 trojk, ki so po meri idf najpogostejše trojke v deklaracijah v izbranih jezikih. Izrišite projekcijo jezikov, kjer vsak jezik predstavite s točko na razsevnem grafu s koordinatama, ki ustrezata prvima dvema komponentama, ter jo označite, kot prikazujem spodaj (zaradi drugega izbora jezikov in procesiranja bo vaš graf drugačen). Na grafu označite, koliko variance v podatkih zajameta prvi dve komponenti PCA. Vse potrebno za izris opravite v funkciji plot_PCA().


Za primerjavo izrišite še projekcijo jezikov v 2D prostoru, ki jo dobite z večrazsežnim lestvičenjem (MDS) s kosinusno razdaljo na celotnih trojkah (vseh terkah dožine 3). Za izračun projekcije MDS uporabite sklearn.manifold.MDS. Vse potrebno za izris opravite v funkciji plot_MDS(). Razlike med izrisom PCA in MDS komentirajte.

Za višje ocene (9 in 10). Za vsakega od 20 jezikov izberite vsaj 5 besedil (torej skupno 100). Nanje aplicirajte PCA, MDS in t-SNE. Pregledno izrišite rezultate ter komentirajte razlike. Uporabite vašo implementacijo PCA. MDS in t-SNE izračunajte s kosinusno razdaljo na vseh terkah dolžine 3; zanju uporabite implementacijo iz sklearn.

Pripravili smo vam ogrodje naloga2.py, ki ga obvezno uporabite. Delovanje implementacije preverite na poljubnih podatkih (priporočam kakšne, zgrajene na roko) ter obvezno tudi na regresijskih testih test_naloga2.py.

Pri PCA boste delali z numpy matrikami. Uporabljajte matrične operacije knjižnice numpy. Pri računanju PCA boste eksplicitno zanko potrebovali zgolj v funkciji "power_iteration", drugje naj jih ne bo. Koristne matrične operacije so denimo mean (povprečje matrike, lahko po stolpcih ali po vrsticah), cov (izračun kovariančne matrike), dot (skalarni produkt ali množenje matrik), zeros, ones (ustvarjanje novih matrik ali vektorjev). Pomagajte si tudi z datoteko numpy_triki.py.

Za risanje diagrama uporabite matplotlib; gotovo vam bodo prav prišle funkcije scatter, text, title.