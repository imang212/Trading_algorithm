# Multi-Asset Trading Algorithm Backtest

## Obsah
- [Multi-Asset Trading Algorithm Backtest](#multi-asset-trading-algorithm-backtest)
  - [Obsah](#obsah)
  - [1. Přehled skriptu](#1-přehled-skriptu)
  - [2. Konfigurace – parametry simulace](#2-konfigurace--parametry-simulace)
    - [Kapitál a náklady](#kapitál-a-náklady)
  - [Profily parametrů dle typu assetu](#profily-parametrů-dle-typu-assetu)
      - [Srovnání parametrů napříč profily](#srovnání-parametrů-napříč-profily)
  - [3. Technické indikátory](#3-technické-indikátory)
    - [3.1 MA Crossover](#31-ma-crossover)
    - [3.2 RSI – Relative Strength Index](#32-rsi--relative-strength-index)
    - [3.3 Bollinger Bands](#33-bollinger-bands)
    - [3.4 MACD](#34-macd)
    - [3.5 ATR – Average True Range](#35-atr--average-true-range)
  - [4. Kombinovaná obchodní strategie](#4-kombinovaná-obchodní-strategie)
    - [BUY signál — skóre ≥ 3 z 5](#buy-signál--skóre--3-z-5)
    - [SELL signál — skóre ≥ 3 z 5](#sell-signál--skóre--3-z-5)
    - [Stop-Loss a řízení pozice](#stop-loss-a-řízení-pozice)
  - [5. Co je to backtest?](#5-co-je-to-backtest)
    - [Průběh backtestу krok za krokem](#průběh-backtestу-krok-za-krokem)
    - [Výkonnostní metriky](#výkonnostní-metriky)
      - [Výnosové metriky](#výnosové-metriky)
      - [Rizikové metriky](#rizikové-metriky)
      - [Metriky kvality obchodů](#metriky-kvality-obchodů)
      - [Skóre indikátorů](#skóre-indikátorů)
  - [6. Výstupy skriptu](#6-výstupy-skriptu)
  - [7. Disclaimer](#7-disclaimer)
   
## 1. Přehled skriptu
Skript provádí historický **backtest kombinované technické obchodní strategie** na devíti různých assetech (zlato, stříbro, akcie). Pro každý asset:
- stáhne historická denní data (OHLCV) přes `yfinance`
- vypočítá pět technických indikátorů
- generuje BUY/SELL signály hlasováním (≥ 3 ze 5 indikátorů musí souhlasit)
- simuluje obchodování včetně poplatků, skluzu a dynamického stop-lossu
- vykreslí grafy a vypíše souhrnnou tabulku metrik

**Rychlý start:**
```bash
pip install yfinance pandas numpy matplotlib tabulate
python trading_backtest.py
```
**Assety ve výchozím nastavení:**

| Asset       | Ticker    | Asset       | Ticker   |
|-------------|-----------|-------------|----------|
| Gold        | GC=F      | Netflix     | NFLX     |
| Silver      | SI=F      | Coca-Cola   | KO       |
| Oil (WTI)   | CL=F      | Agnico Eagle| AEM      |
| USD Index   | DX-Y.NYB  | Novo Nordisk| NVO      |
| MSFT        | MSFT      | Moneta      | MONET.PR |
| GOOGL       | GOOGL     | ORCL        | ORCL     |
| NVDA        | NVDA      | AMD         | AMD      |

## 2. Konfigurace – parametry simulace
Konstanty na začátku souboru ovlivňují realismus a přísnost simulace.

### Kapitál a náklady
| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `INITIAL_CAP` | `10 000` USD | Počáteční kapitál přidělený **každému assetu zvlášť**. Celková simulace pracuje s 9 × $10 000 = $90 000. |
| `COMMISSION` | `0.001` (0,1 %) | Poplatek brokera za jeden obchod. Odečítá se při každém nákupu i prodeji. Reálné poplatky: CFD 0,05–0,2 %, akcie i méně. |
| `SLIPPAGE` | `0.0005` (0,05 %) | Skluz – rozdíl mezi požadovanou a skutečně dosaženou cenou. Při nákupu se cena zvýší, při prodeji sníží. Závisí na likviditě trhu. |

## Profily parametrů dle typu assetu
Různé typy assetů mají různou volatilitu a tržní charakter, proto každý asset dostane vlastní sadu parametrů indikátorů. Skript definuje čtyři profily:

| Profil | Assety | Filozofie |
|--------|--------|-----------|
| `COMMODITY` | Gold, Silver, Oil | Silné trendy, vysoká volatilita → pomalejší MA, širší BB pásma, větší ATR stop-loss |
| `FOREX_IDX` | USD Index | Velmi pomalé pohyby → nejpomalejší MA, úzká BB pásma, nejmenší ATR multiplikátor |
| `TECH` | MSFT, GOOGL, NVDA, AMD, Netflix, Spotify, ORCL | Vysoká beta, rychlé trendy → standardní agresivní nastavení |
| `DEFENSIVE` | Coca-Cola, Novo Nordisk, Agnico Eagle, Moneta | Nízká beta, dividendové akcie → pomalejší indikátory, konzervativní stop-loss |


#### Srovnání parametrů napříč profily

| Parametr | COMMODITY | FOREX_IDX | TECH | DEFENSIVE | Proč se liší |
|----------|-----------|-----------|------|-----------|-------------|
| `MA_SHORT` | 30 | 40 | 20 | 25 | Volatilnější asset → delší MA pro filtraci šumu |
| `MA_LONG` | 75 | 100 | 50 | 60 | Komodity mají delší cykly než tech akcie |
| `RSI_PERIOD` | 14 | 21 | 14 | 14 | USDIDX se pohybuje pomalu → delší perioda |
| `RSI_OB` | 70 | 65 | 70 | 65 | Defenzivní/forex méně extrémní výkyvy |
| `RSI_OS` | 30 | 35 | 30 | 35 | Defenzivní/forex méně extrémní výkyvy |
| `BB_PERIOD` | 25 | 30 | 20 | 20 | Delší perioda = klidnější pásma u pomalejších assetů |
| `BB_STD` | 2.5 | 1.8 | 2.0 | 1.8 | Komodity: širší pásma pro vysokou volatilitu |
| `MACD_SLOW` | 30 | 35 | 26 | 26 | Pomalejší MACD u komodit a forexu |
| `ATR_SL_MULT` | 2.5 | 1.5 | 2.0 | 1.8 | Komodity: větší stop-loss kvůli šumu; forex: užší |

## 3. Technické indikátory
Skript kombinuje pět indikátorů, z nichž každý sleduje jinou vlastnost trhu: **trend, hybnost a volatilitu**. Jejich kombinace snižuje počet falešných signálů.

### 3.1 MA Crossover
**Moving Average Crossover** (křížení klouzavých průměrů) ukazuje, zda je krátkodobý trend nad nebo pod dlouhodobým.

Skript používá **EMA** (Exponential Moving Average), která přikládá vyšší váhu novějším datům, a **SMA** (Simple Moving Average) pro vizualizaci.

```
EMA(t) = α × cena(t) + (1 − α) × EMA(t−1)
kde  α = 2 / (perioda + 1)
```

| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `MA_SHORT` | `20` | Počet dní pro krátký klouzavý průměr (EMA i SMA). Citlivý, reaguje rychle na pohyby ceny. |
| `MA_LONG` | `50` | Počet dní pro dlouhý klouzavý průměr. Pomalejší, filtruje krátkodobý šum. |

**Signál:**
- `EMA_short > EMA_long` → krátkodobý trend roste nad dlouhodobý → **BUY skóre**
- `EMA_short < EMA_long` → krátkodobý trend klesl pod dlouhodobý → **SELL skóre**
  
**SMA 20 (krátkodobý trend):**

Průměr posledních 20 obchodních dní = přibližně 1 měsíc. Říká ti, jak si vede cena v krátkodobém horizontu.
- Cena nad SMA20 → krátkodobý momentum je bullish, trh roste
- Cena pod SMA20 → krátkodobý momentum je bearish, trh klesá
- SMA20 funguje jako dynamická podpora/odpor – cena se k němu často vrací

**SMA 50 (střednědobý trend):**

Průměr posledních 50 obchodních dní = přibližně 2.5 měsíce. Sledují ho hlavně institucionální investoři.
- Cena nad SMA50 → střednědobý trend je zdravý
- Cena pod SMA50 → trh je ve střednědobém downtrend
- Prolomení SMA50 je považováno za silnější signál než SMA20

**Golden Cross a Death Cross:**

To je to nejdůležitější na kombinaci SMA20/SMA50:
- Golden Cross:  SMA20 překříží SMA50 ZDOLA NAHORU  → silný BUY signál
- Death Cross:   SMA20 překříží SMA50 SHORA DOLŮ    → silný SELL signál

### 3.2 RSI – Relative Strength Index
**RSI** měří sílu a hybnost cenových pohybů na škále 0–100. Porovnává průměrné zisky a průměrné ztráty za sledované období a identifikuje zóny překoupenosti nebo přeprodanosti.

```
RSI = 100 − [100 / (1 + RS)]
kde  RS = průměrný zisk za N dní / průměrná ztráta za N dní
```

| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `RSI_PERIOD` | `14` | Počet dní pro výpočet RSI. Standardní hodnota dle J. Wellese Wildera (tvůrce RSI). |
| `RSI_OB` | `70` | Hranice překoupenosti (overbought). Nad touto hodnotou je trh „drahý" – možný obrat dolů. |
| `RSI_OS` | `30` | Hranice přeprodanosti (oversold). Pod touto hodnotou je trh „levný" – možný odraz nahoru. |

**Signál (ve strategii):**
- `RSI < 50` → cena ještě nerostla příliš rychle, prostor k růstu → **BUY skóre**
- `RSI > 50` → cena roste rychle, hybnost může slábnout → **SELL skóre**

> Poznámka: hranice 70/30 slouží pro vizualizaci v grafu, strategie používá střední linii 50.
>
**Co hledat:**
- RSI klesne pod 30 → trh je přeprodaný, cena klesla příliš rychle → potenciální odraz nahoru → hledáš BUY
- RSI vystoupí nad 70 → trh je překoupený, cena rostla příliš rychle → potenciální obrat dolů → hledáš SELL
- RSI překříží linku 50 zdola nahoru → potvrzení rostoucího trendu → BUY signál
- RSI překříží linku 50 shora dolů → potvrzení klesajícího trendu → SELL signál

### 3.3 Bollinger Bands
**Bollingerova pásma** se skládají ze tří čar: středového klouzavého průměru (SMA) a dvou pásem vzdálených N násobků směrodatné odchylky (σ). Pásma se rozšiřují při vysoké volatilitě a zužují při nízké.

```
BB_upper = SMA(N) + k × σ(N)
BB_lower = SMA(N) − k × σ(N)
BB_pct   = (cena − BB_lower) / (BB_upper − BB_lower)   →  hodnota 0–1
```

| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `BB_PERIOD` | `20` | Počet dní pro výpočet středového SMA a směrodatné odchylky. |
| `BB_STD` | `2.0` | Multiplikátor σ. Při 2σ je statisticky uvnitř pásem ~95 % cen (normální rozdělení). |

**Signál:**
- `BB_pct < 0.4` → cena blízko dolního pásma = potenciálně podhodnocená → **BUY skóre**
- `BB_pct > 0.6` → cena blízko horního pásma = potenciálně předražená → **SELL skóre**

**Co hledat:**
- Cena se dotkne dolního pásma → statisticky podhodnocená → potenciální BUY
- Cena se dotkne horního pásma → statisticky předražená → potenciální SELL
- Pásma se zužují (squeeze) → trh je klidný, hromadí se energie → brzy přijde velký pohyb, ale nevíš jakým směrem
- Pásma se rozšiřují → trh je ve volatile fázi, trend je silný
- Cena "jede podél horního pásma" → velmi silný uptend, nesell příliš brzy
- BB% (ve skriptu) = 0 znamená cena na dolním pásmu, 1 = na horním pásmu, 0.5 = uprostřed

### 3.4 MACD
**MACD** (Moving Average Convergence/Divergence) měří hybnost a sílu trendu porovnáním dvou exponenciálních klouzavých průměrů.

```
MACD      = EMA(12) − EMA(26)
Signal    = EMA(9) aplikovaná na MACD
Histogram = MACD − Signal
```

| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `MACD_FAST` | `12` | Počet dní pro rychlou EMA. Kratší = citlivější na pohyby ceny. |
| `MACD_SLOW` | `26` | Počet dní pro pomalou EMA. Delší = zachytí střednědobý trend. |
| `MACD_SIGNAL` | `9` | Délka signal linie (EMA nad MACD). Standardní hodnota ze světových burz. |

**Signál:**
- `MACD > Signal` → rostoucí hybnost → **BUY skóre**
- `MACD < Signal` → klesající hybnost → **SELL skóre**

**Co hledat:**
- MACD překříží Signal zdola nahoru → BUY signál (hybnost roste)
- MACD překříží Signal shora dolů → SELL signál (hybnost klesá)
- Histogram roste → trend sílí, drž pozici
- Histogram se zmenšuje → trend slábne, připrav se na výstup
- MACD nad nulou → celkový trend je bullish
- MACD pod nulou → celkový trend je bearish

Zlaté pravidlo: MACD je lagging indikátor – potvrzuje trend, který už začal. Nikdy nebude signalizovat úplně na vrcholu ani dně, ale filtruje falešné pohyby.

### 3.5 ATR – Average True Range
**ATR** měří průměrnou denní volatilitu (rozkmit ceny). Na rozdíl od ostatních indikátorů ATR **neříká směr** – říká, o kolik se cena typicky pohybuje za den. Ve skriptu slouží primárně pro nastavení dynamického stop-lossu.

```
True Range(t) = max(High − Low, |High − Close(t−1)|, |Low − Close(t−1)|)
ATR(N) = průměr True Range za posledních N dní
Stop-Loss = vstupní_cena − ATR_SL_MULT × ATR
```

| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `ATR_PERIOD` | `14` | Počet dní pro průměrování True Range. Standardní hodnota doporučená tvůrcem Wilderem. |
| `ATR_SL_MULT` | `2.0` | Multiplikátor ATR pro stop-loss. Hodnota 2 = stop-loss je 2× průměrný denní rozkmit pod vstupem. Nižší → agresivnější (více stop-lossů). Vyšší → konzervativnější. |

**Co hledat:**
- Vysoký ATR → trh je nervózní, velké pohyby → stop-loss musí být vzdálenější, jinak tě vyhodí šum
- Nízký ATR → klidný trh → stop-loss může být blíže na vstupní ceně, risk je menší
- ATR náhle skočí nahoru → přišla velká zpráva (earnings, FED, geopolitika) → pozor na zvýšené riziko
- ATR dlouhodobě klesá → trh usíná, blíží se velký pohyb (podobně jako BB squeeze)

## 4. Kombinovaná obchodní strategie
Klíčovým rysem je, že **žádný indikátor sám o sobě nevydá signál**. Každý indikátor přispěje ke skóre BUY nebo SELL, a obchod nastane až tehdy, když **alespoň 3 ze 5 indikátorů souhlasí**. Tím se výrazně snižuje počet falešných signálů.

### BUY signál — skóre ≥ 3 z 5
| # | Podmínka | Co říká |
|---|----------|---------|
| 1 | `EMA_short (20) > EMA_long (50)` | Krátkodobý trend je nad dlouhodobým → bullish |
| 2 | `RSI < 50` | Cena ještě nerostla příliš rychle, je prostor k dalšímu růstu |
| 3 | `BB_pct < 0.4` | Cena blízko dolního pásma → potenciálně podhodnocená |
| 4 | `MACD > Signal` | Rychlá linie překřížila pomalou nahoru → rostoucí hybnost |
| 5 | `cena > SMA_short` | Cena je nad krátkodobým průměrem → pohyb s trendem |

### SELL signál — skóre ≥ 3 z 5
| # | Podmínka | Co říká |
|---|----------|---------|
| 1 | `EMA_short (20) < EMA_long (50)` | Krátkodobý trend klesl pod dlouhodobý → bearish |
| 2 | `RSI > 50` | Cena roste rychle, hybnost může slábnout |
| 3 | `BB_pct > 0.6` | Cena blízko horního pásma → potenciálně předražená |
| 4 | `MACD < Signal` | Rychlá linie klesla pod pomalou → klesající hybnost |
| 5 | `cena < SMA_short` | Cena je pod krátkodobým průměrem → ztráta trendu |

### Stop-Loss a řízení pozice
Při každém BUY je automaticky nastaven dynamický stop-loss:

```
stop_loss = vstupní_cena − ATR_SL_MULT × ATR(14)
```
Pokud cena klesne pod tuto úroveň dříve, než přijde SELL signál, pozice se okamžitě uzavře jako **STOP-LOSS** obchod a omezí tak maximální ztrátu. Do každého obchodu je investováno **95 % dostupného kapitálu** (5 % zůstává jako rezerva na poplatky a skluz).

<img width="3585" height="1359" alt="signals" src="https://github.com/user-attachments/assets/60171474-3dac-47c1-9aaf-584c3b517be1" />

## 5. Co je to backtest?
Backtest je **historická simulace obchodní strategie**. Místo skutečného obchodování skript prochází historická data den po dni a aplikuje strategii, jako by se obchodovalo v reálném čase — ale na minulých datech.

### Průběh backtestу krok za krokem
1. **Stažení dat** — `yfinance` stáhne denní OHLCV data od `2021-01-01` do dnes.
2. **Výpočet indikátorů** — pro každý den se vypočítají hodnoty všech pěti indikátorů.
3. **Generování signálů** — pro každý den se spočítá BUY a SELL skóre (0–5). Pokud ≥ 3 → signál.
4. **Simulace obchodů** — skript prochází data řádek po řádku a simuluje nákupy a prodeje včetně poplatků a skluzu.
5. **Sledování kapitálu** — po každém dni se zaznamená celková hodnota portfolia (cash + otevřená pozice).
6. **Výpočet metrik** — po skončení simulace se vypočítají výkonnostní ukazatele.

### Výkonnostní metriky
#### Výnosové metriky

| Metrika | Vzorec | Interpretace |
|---------|--------|--------------|
| **Celkový výnos (%)** | `(finální_kapitál − počáteční_kapitál) / počáteční_kapitál × 100` | Kolik procent vydělala nebo prodělala strategie za celé sledované období. Např. +39 % znamená, že z $10 000 se stalo $13 900. Nezohledňuje čas ani riziko. |
| **Buy & Hold (B&H)** | `(cena_konec − cena_začátek) / cena_začátek × 100` | Benchmark — kolik by vydělal investor, který by jednoduše nakoupil na začátku a prodal na konci bez jakéhokoli obchodování. Slouží jako referenční bod: pokud strategii B&H nepřekoná, není obchodování lepší než prostý nákup a držení. |
| **Alpha** | `výnos_strategie − výnos_B&H` | Přidaná hodnota aktivního obchodování oproti pasivnímu držení. **Kladná alpha** = strategie překonala trh. **Záporná alpha** = bylo by lepší jen nakoupit a držet. Např. alpha −120 % u NVDA znamená, že B&H by bylo o 120 % lepší než strategie. |

#### Rizikové metriky

| Metrika | Vzorec | Interpretace |
|---------|--------|--------------|
| **Sharpe Ratio** | `průměrný_denní_výnos / std_denních_výnosů × √252` | Měří výnos *vzhledem k podstupovanému riziku* (volatilitě). Čím vyšší, tím lépe. `√252` převádí denní hodnoty na roční. **< 0** = strategie prodělává. **0–1** = slabý výkon. **1–2** = dobrý výkon. **> 2** = výborný výkon. Příklad: Sharpe 1.56 u Moneta znamená solidní rizikově vážený výnos. |
| **Max Drawdown (%)** | `min((equity − rolling_max) / rolling_max × 100)` | Největší pokles hodnoty portfolia od historického vrcholu ke dnu — kdykoli v průběhu celého období. Odpovídá na otázku: *„V nejhorším scénáři, o kolik bych přišel?"* Např. Max DD −58 % znamená, že v nejhorším bodě ztratila strategie 58 % hodnoty od svého vrcholu. Nižší (méně záporné) = méně rizikové. |

#### Metriky kvality obchodů

| Metrika | Vzorec | Interpretace |
|---------|--------|--------------|
| **Win Rate (%)** | `počet_ziskových_obchodů / celkový_počet_obchodů × 100` | Procento obchodů, které skončily ziskem. **Pozor:** vysoký win rate sám o sobě nestačí — strategie s win rate 40 % může být zisková, pokud průměrný zisk výrazně převyšuje průměrnou ztrátu. Ideálně sledovat společně s Profit Factor. |
| **Profit Factor** | `součet_všech_zisků / součet_všech_ztrát` | Kolik korun zisku přinese každá koruna ztráty. **< 1** = strategie celkově prodělává. **1–1.5** = mírně zisková. **1.5–2** = dobrá strategie. **> 2** = velmi dobrá strategie. Např. Profit Factor 4.22 u Moneta znamená, že na každý $1 ztráty připadají $4.22 zisku. |
| **# BUY** | počet | Kolikrát algoritmus vydal BUY signál a otevřel pozici. |
| **# SELL** | počet | Kolikrát byl obchod uzavřen normálním SELL signálem (plánovaný výstup). |
| **# STOP** | počet | Kolikrát byl aktivován stop-loss (nucený výstup při poklesu pod limit). Vysoký poměr STOP/BUY signalizuje špatné načasování vstupů nebo příliš těsný stop-loss. |

#### Skóre indikátorů

| Metrika | Rozsah | Interpretace |
|---------|--------|--------------|
| **Ø BUY score** | 0.0 – 5.0 | Průměrný počet bullish indikátorů za celé sledované období (průměr přes všechny dny). Hodnota blízká 5 = trh byl po většinu doby v silném rostoucím trendu. Hodnota blízká 2.5 = neutrální trh bez jasného směru. |
| **Ø SELL score** | 0.0 – 5.0 | Průměrný počet bearish indikátorů za celé sledované období. Součet `Ø BUY score + Ø SELL score` nemusí dávat 5, protože každý indikátor se hodnotí nezávisle pro BUY i SELL podmínku. Vysoký SELL score = trh byl převážně v klesajícím trendu. |

<img width="2685" height="1110" alt="summary_table" src="https://github.com/user-attachments/assets/a36c4d8f-70d3-4c1d-a611-e5473367bb30" />

<img width="2385" height="1479" alt="summary_comparison" src="https://github.com/user-attachments/assets/f3a9562b-0f93-42f3-a489-983caafd65be" />

## 6. Výstupy skriptu
Po dokončení backtestу skript vygeneruje:
- **Terminál** — přehledná tabulka se všemi metrikami pro každý asset.
- **`chart_<asset>.png`** — individuální 5-panelový graf pro každý asset:
  - Panel 1: Cena + SMA + Bollinger Bands + označené BUY/SELL/STOP obchody
  - Panel 2: Equity křivka (vývoj kapitálu v čase)
  - Panel 3: RSI s vyznačenými zónami překoupenosti/přeprodanosti
  - Panel 4: MACD s histogramem
  - Panel 5: ATR (volatilita)
- **`summary_comparison.png`** — souhrnný srovnávací graf všech assetů (výnosy vs B&H, Sharpe Ratio, Max Drawdown, Win Rate).

## 7. Disclaimer
> **Upozornění:** Tento skript slouží výhradně k **vzdělávacím a analytickým účelům**.
> Backtest na historických datech **nezaručuje budoucí výsledky**.
> Výkonnost v minulosti ≠ výkonnost v budoucnosti.
> Skript **nepředstavuje investiční poradenství**.
> Před reálným obchodováním konzultujte licencovaného finančního poradce.
