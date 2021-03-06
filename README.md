# Szuperrezolúció hajókon


Kruppai Gábor<br/>
_kruppaigabor gmail com_<br/>
(A180FM)

Szabó Dávid<br/>
_david gmail com_<br/>
(JLF6V4)

Kürty Gyula László<br/>
(I492q2)<br/>
**Bővített leírás:**
[GoogleDocs](https://docs.google.com/document/d/1boMNZsrCYYDBOrZ-RbZ02iLHjTWZAesnbHT8jQKZdCE/view)


### Feladat leírása
> * Szerezz 5000 db legalább 64x64 méretű képet, amin hajó van. Használhatod ehhez például a google images-t (alább egy releváns link), vagy az ImageNet adathalmazt.
> * Készíts az összes képből több változatot, a következő méretekben: 64x64, 32x32. (Ha nem 1:1 arányú a kép, vágd ki a közepét.)
> * Készíts egy neurális hálózatot, ami az 32x32 pixeles képből megtanulja visszaállítani a 64x64 pixeles képet!

### Forráskód
[Github](https://github.com/gkrupp/Deep-Learning-SuperResolution)

### Adathalmaz
[GoogleDrive](https://drive.google.com/file/d/1VI4INN6nyRiIN5Wu9Fpj8Euu455PSnz5)

### Futtatás
[run.md](https://github.com/gkrupp/Deep-Learning-SuperResolution/blob/master/run.md)

---


#### Képek beszerzése
Internetes keresés után több opció is felmerült a szűkséges adatok összegyűjtésére: *GoogleImages*, *CIFAR10*, *ImageNet*.

A **CIFAR10** egy 10 különböző kategóriát - köztük hajókat is - tartalmazó `32x32` méretű képekből álló adathalmaz. Kategóriánként 5000 training képpel és 1000 validation képpel. Ezt előkészítve kaptuk volna, azonban az elérhető maximális képméret 32x32 volt, így a `64x64`-es követelmény miatt mást kellett választani.

A **GoogleImages** szintén jó forrás és viszonylag sok és jól dokumentált eszköz elérhető a képek letöltésére, ám a képek saját megoldással való gyűjtése elég időigényes, valamint a Google-ben megjelenő képek nem minden esetben kapcsolódnak a keresett kifejezéshez, ezért ez nem túl megbízható forrás. Minden esetre érdemes észben tartani ezt az opciót is esetleges későbbi adatszerzéshez.

A végső megoldás az **ImageNet** használata volt. Az ImageNet egy, a WordNet struktúrájára épülő kép-adatbázis, azaz a WordNet fa gráfjának minden csúcsa kategóriaként tekinthető amiben képek szerepelnek. Ez a forrás biztosan a kategóriákhoz tartozó képeket tartalmazza, mivel az itteni képeket emberek osztották a kategóriáikba. Kutatási és oktatási célokra - indoklás és egyetemi tagság igazolása után - szabadon elérhető az összes kép (csak egyetemi célokra). Ez a forrás tűnt a legbiztosabbnak és legegyszerűbbnek, ezért a képek innen lettek beszerezve.

#### Képek letöltése
Először a **WordNet** segítségével meg lett határozva a legmagasabban lévő hajóhoz kapcsolódó szó: `vessel`. Ez alapján már lehetett szűrni ImageNet-en is, ám letölteni csak egy kategóriát lehet egyszerre, azaz például csak a pontosan `vessel` szóhoz tartozókat, tehát a csúcs gyerekei - speciálisabb szavak - ekkor nem kerülnek bele a halmazba. Szerencsére az ImageNet lehetőséget kínál egy csúcspont összes leszármazottjának kilistázására (azonosítók formájában).
A letöltéseket - ismételt jogosultság kéréssel - ImageNet API-kon keresztül is végre lehetett hajtani, így automatizálva a folyamatot. Adott API-kulcs és kategória lista (~180 alkategória) lévén, már nem volt nehéz URL-eket generálni és letölteni az összes képet. (ezt a feladatot egy JavaScript kód végezte)
Összességében több, mint 121 ezer - biztosan hajókat tartalmazó - képet sikerült szerezni, enyhén felélőve a kitűzött darabszámnak.

#### Képek átméretezése
A letöltött képeknél elég nagy volt a felbontás- és formátumbeli diverzitás. Az “előkészítés” során először ki lettek dobva azok a képek, amelyek kisebbik oldal nem érte el a 64 pixelt, így végül `120'804` db maradt. Utána az összes kép konvertálva lett az **ImageMagick** program segítségével a lehető legjobb minőségű jpeg formátumba `64x64` és `32x32` felbontásokba, méretarányosan kicsinyítve a képet a rövidebb oldal hosszát figyelembe véve, majd kivágva a kép közepi `64x64` / `32x32` pixeles részt.
Ezután még elő kellett állítani a `32x32`-ből köbös interpolációval `64x64`-es méretbe konvertált képeket, mivel a tanításhoz használt módszernél azokat kellett használni.

#### Tanítóhalmaz készítése
A tanításhoz a képeket át kellett alakítani 3D mátrixokká, pixeleik RGB értékeit `[0,1]` intervallumra kellett hozni és csak az `RGB` csatornákat megtartani (vagy fekete-fehér esetben előállítani). így a képek egységes “formátumba” kerültek és az azonos méretűeket össze lehetett fűzni egy 4 dimenziós tenzorrá: `T[picture][layer][x][y]`. Ekkor kaptunk három tenzort: `32x32`, `64x64` és `64x64` átméretezett. Ezek egy `hdf5` fájlban lettek eltárolva. Megjegyzendő, hogy itt már előjött egy, a témában gyakori probléma: a tanítóhalmaz már előkészítésnél sem fért bele a számunkra elérhető memóriába, így a képeket blokkonként kellett feldolgozni és fájlba írni. (ezekre a feladatokra egy Python script lett írva)

#### Adatgyűjtés végeredménye
A teljes tanítóhalmaz `12.4GB` lett, ami tartalmazza a `32x32`, `64x64` és az átméretezett `64x64lanczos` képeket, valamint az eredeti képek listáját sorrendben (a visszakereshetőség végett). A nagy méret miatt a teljes fájl nem tudjuk online elérhetővé tenni, ezért az ebből készült `4GB`-os, `40'000` képet tartalmazó kivonat kerül megosztásra.
A letöltéshez és a feldolgozáshoz használt scriptek és segédfájlok elérhetőek a `datproc/` mappában. (Mivel nem az  adatgyűjtés volt az elsődleges cél, ezért az ehhez kapcsolódó fájlok nincsenek dokumentálva.)

#### Szuperrezolúció
A képeken értelmezett szuperrezolúció azt a problémát igyekszik megoldani, hogy egy kis felbontású rasztergrafikus képből egy nagyobb felbontású rasztergrafikus képet lehessen csinálni olyan módon, hogy a hiányzó részeket “kitalálja”. Többféle módszer létezik, az egyik legegyszerűbb megoldáscsoport az interpolációk, amit a grafika területén alkalmaznak előszerettel (pl.: lineáris, négyzetes interpoláció, esetleg spline-ok alkalmazása lásd: OpenGl) előnye, hogy gyorsan számolható, optimalizálható. 
Mivel úgy tűnik, hogy egy-egy pixelt erősen meghatároz a környezete, vagyis a pixel színe nem független a környező pixelek színeitől, intuitívan alkalmazhatónak tűnnek rá a deep learning algoritmusok. És mivel feltételezhetjük, hogy egy új pixel színe sokkal kevésbé függ egy távoli pixel színétől és sokkal inkább egy közelitől, intuitívan azt is beláthatjuk, hogy valószínűleg fontosak lesznek a konvolúciós layerek számunkra. Azért a konvolúciós layerek, mert a konvolúciós layerek esetén explicit szerepel az, hogy a kimenet egy pixelét az előző rétegben szereplő pixelek egy közeli csoportja segítségével határozzuk meg. Persze egy teljesen kapcsolt előrecsatolt neurális hálózat is képes lenne megtanulni a probléma megoldását mégis célszerűnek tűnik konvolúciós rétegek alkalmazása, hiszen ha a hipotézisünk igaznak bizonyul és egy új pixel színe tényleg a környezetétől függ inkább, abban az esetben a teljesen kapcsolt feedforward networkünk két rétege között a súlyok nagy része a nullához konvergálna így érdemes lenne a felesleges számítási kapacitást megspórolni, illetve a túltanulást elkerülni azzal, hogy inkább rögtön a konvolúciós rétegeket használjuk. Természetesen nem csak a hagyományos konvolúciót értem a konvolúciós rétegek alatt, hanem az úgynevezett “upconvolution” fajtákat is amik képesek egy adott bemeneti tensorra egy magasabbat és szélesebbet is gyártani. (ez főleg fontos figyelembe véve azt, hogy a szuperrezolúciónál nekünk a kimeneten egy kétszer nagyobb képet kell eredményül kapnunk)
Az egyik legegyszerűbb és legnépszerűbb installáció a feladat megoldására az SRCNN amely csupán három konvolúciós réteget tartalmaz. A layerek meghatározásai: patch extrakció (patch extraction, sajnos nem tudok rá jobb magyar kifejezést) és reprezentáció, nem lineáris leképezés (linearitás elkerülése végett) és rekonstrukció. Egy másik kicsit felül teljesítő bevált módszer az FSRCNN. Az összehasonlításukra az alábbi szemléletes illusztrációt találtam:

![SRCNN - FSRCNN](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN/img/framework.png)
> _Source: [`http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html`](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)_
