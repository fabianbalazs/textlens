# TextLens

Egy nagyon egyszerű, saját készítésű szöveges keresőmotor Pythonban.
Egyetemi NLP kurzushoz készült, ezért a cél az érthetőség.

**Verzió: v0.1.1**

## Mit tud?

- beolvassa a `documents/` mappa `.txt` fájljait
- tisztítás, tokenizáció, stopwordök (v0.1)
- szógyakoriság és logaritmikus súlyozás (v0.1, `--log` móddal)
- **v0.1.1:** vektorizáció (indexelés, one-hot, Bag of Words) és **TF-IDF alapú rangsorolás**

## Telepítés és futtatás

```bash
pip install -r requirements.txt    # csak a scikit-learn kell

python app.py            # TF-IDF rangsor (alapértelmezett)
python app.py --log      # v0.1 logaritmikus súlyozás
```

## Példa

```
TEXTLENS
4 dokumentum betöltve, mód: TF-IDF. (Kilépés: üres sor)

Keresés: python machine learning

1. machine_learning.txt
   Relevancia: 1.18

2. python.txt
   Relevancia: 0.49

3. web_development.txt
   Relevancia: 0.10
```

Ha nincs találat: `Nem található releváns dokumentum.`

## Hogyan működik?

**Közös alap:** minden szöveg átmegy a `preprocess` függvényen (kisbetűsítés, írásjelek és felesleges whitespace törlése, szavakra bontás, stopwordök kiszűrése).

**Vektorizáció** (`textlens/vectorization.py`)
1. *Indexelés:* minden szóhoz egy szám (`{szó: index}`), ismeretlen szó = `-1`.
2. *One-hot:* egy szó egy csupa 0 vektor, egyetlen 1-essel a szó indexénél.
3. *Bag of Words:* egy dokumentum = a szókincs minden szavának darabszáma (`CountVectorizer`).

**TF-IDF**
- **TF:** a szó darabszáma osztva a dokumentum szavainak számával.
- **IDF:** minél kevesebb dokumentumban szerepel egy szó, annál nagyobb az értéke, így a ritka szó többet ér.
- **TF-IDF = TF × IDF.**

**Pontszám:** a keresett szavak TF-IDF értékeinek összege az adott dokumentumban. A szókincsben nem szereplő keresőszavakat figyelmen kívül hagyjuk, és csak a legalább egy találatot tartalmazó dokumentumok kerülnek a listára.

> Megjegyzés: a scikit-learn a tankönyvi `log(D/d)` képletet két ponton módosítja: simított IDF-et használ (`ln((1+D)/(1+d)) + 1`), és a dokumentumvektorokat L2 normalizálja. Emiatt a szám nem egyezik pontosan a kézi számolással, a rangsor logikája viszont ugyanaz.

## Projektstruktúra

```
TextLens/
├── documents/       # a keresett .txt fájlok
├── app.py           # minden logika: tisztítás, vektorizáció, keresés, main()
├── requirements.txt
└── README.md
```

Csak az `app.py`-t kell futtatnod, minden más függvény ebből van meghívva.

## Roadmap

**v0.1**
- text cleaning
- tokenization
- stopwords
- word frequency
- logarithmic weighting
- basic document search

**v0.1.1**
- vectorization (indexing, one-hot, bag of words)
- TF, IDF, TF-IDF
- TF-IDF based document search

