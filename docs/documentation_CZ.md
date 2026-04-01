# Multi-Asset Trading Algorithm
## Obsah
- [Multi-Asset Trading Algorithm](#multi-asset-trading-algorithm)
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
  - [5. Backtesting a Metriky výkonnosti](#5-backtesting-a-metriky-výkonnosti)
    - [Průběh backtestу krok za krokem](#průběh-backtestу-krok-za-krokem)
    - [Výkonnostní metriky](#výkonnostní-metriky)
      - [Výnosové metriky](#výnosové-metriky)
      - [Rizikové metriky](#rizikové-metriky)
      - [Metriky kvality obchodů](#metriky-kvality-obchodů)
      - [Skóre indikátorů](#skóre-indikátorů)
  - [6. Návod k použití / Režimy (Bash Script)](#6-návod-k-použití--režimy)
  - [7. Analýza rychlosti a objemu](#7-analýza-rychlosti-a-objemu)
    - [7.1 Rate of Change (ROC)](#71-rate-of-change-roc)
    - [7.2 ATR trend](#72-atr-trend)
    - [7.3 Candle body size](#73-candle-body-size)
    - [7.4 Volume](#74-volume)
    - [7.5 Volume trend](#75-volume-trend)
    - [7.6 OBV divergence (On-Balance Volume)](#76-obv-divergence-on-balance-volume)
    - [7.7 Jak kombinovat všech 6 metrik](#77-jak-kombinovat-všech-6-metrik)
  - [8. Monte Carlo predikce](#8-monte-carlo-predikce)
  - [9. Prophet predikce](#9-prophet-predikce) 
  - [10. Streamlit Dashboard](#10-streamlit-dashboard)  
  - [11. Disclaimer](#11-disclaimer)
   
## 1. Přehled skriptu
Skript provádí historický **backtest kombinované technické obchodní strategie** na devíti různých assetech (zlato, stříbro, akcie). Pro každý asset:
- stáhne historická denní data (OHLCV) přes `yfinance`
- vypočítá pět technických indikátorů
- generuje BUY/SELL signály hlasováním (≥ 3 ze 5 indikátorů musí souhlasit)
- simuluje obchodování včetně poplatků, skluzu a dynamického stop-lossu
- vykreslí grafy a vypíše souhrnnou tabulku metrik

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
| ...        | ...      | ...         | ...      |


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


### Srovnání parametrů napříč profily

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

<img width="3585" height="1894" alt="signals_table" src="https://github.com/user-attachments/assets/44c4eda1-440f-49da-b74e-200ccce1e0a9" />

## 5. Backtesting a Metriky výkonnosti
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

<img width="2685" height="1542" alt="summary_table" src="https://github.com/user-attachments/assets/995add8e-a4cc-4e52-870a-64368c848854" />

<img width="2385" height="1478" alt="summary_comparison" src="https://github.com/user-attachments/assets/42e17b7d-2ca5-4d47-9c52-08d694acbb81" />


## 6. Návod k použití / Režimy 
### Požadavky
- python: 3.9+
```bash
pip install yfinance pandas numpy matplotlib tabulate prophet
```

### Režim 1 – Plný backtest (výchozí)
Spustí backtest pro všechny assety od `START_DATE` do dnes, vygeneruje grafy a tabulky (historických denních dat).
```bash
python trading_backtest.py
```

**Co se stane:**
1. Stáhne historická denní data pro všech assetů (může trvat 1–3 minuty)
2. Pro každý asset spočítá indikátory a simuluje obchodování
3. Vytiskne souhrnnou tabulku do terminálu
4. Vytiskne roční breakdown (výnos / win rate / Sharpe po letech)
5. Vytiskne aktuální signály a cenové hladiny
6. Uloží PNG soubory (grafy, tabulky, signály, příkazy)

**Výstupy skriptu po dokončení backtestů:**
- **Terminál** — několik sekcí výstupu:
  - Souhrnná tabulka metrik pro každý asset
  - Roční breakdown (výnos / win rate / Sharpe po letech)
  - Aktuální signály a cenové hladiny
  - Doporučené příkazy (Buy Limit, Stop-Loss, Take Profit)
- **`chart_<asset>.png`** — individuální graf pro každý asset se dvěma sloupci:
  - Levý sloupec (celé období): Cena + MA + BB + obchody + Monte Carlo predikce, Equity křivka, RSI, MACD, ATR
  - Pravý sloupec (zoom posledních 6 měsíců): stejné panely přiblížené na konec dat
- **`summary_comparison.png`** — souhrnný srovnávací graf všech assetů (výnosy vs B&H, Sharpe Ratio, Max Drawdown, Win Rate)
- **`summary_table.png`** — souhrnná tabulka všech metrik jako PNG
- **`signals.png`** — tabulka aktuálních signálů pro všechny assety
- **`order_levels.png`** — tabulka doporučených cenových příkazů

### Režim 2 – Rychlá analýza jednoho assetu
Stáhne data jen pro jeden asset a okamžitě zobrazí aktuální stav indikátorů a cenové hladiny. Trvá 5–10 sekund.
```bash
python trading_backtest.py --analyze 
python trading_backtest.py --analyze  --interval 
```

**Příklady:**
```bash
# Denní data (výchozí)
python trading_backtest.py --analyze Gold
# Hodinová data
python trading_backtest.py --analyze NVDA --interval 1h
# 4hodinová data (resampleno z hodinových)
python trading_backtest.py --analyze Bitcoin --interval 4h
# 15minutová data
python trading_backtest.py --analyze Spotify --interval 15m
```

**Dostupné intervaly:**
| Interval | Historie | Vhodné pro |
|----------|----------|------------|
| `1m` | 5 dní | Velmi krátkodobé signály |
| `5m` | 30 dní | Intraday swingtrading |
| `15m` | 30 dní | Intraday analýza |
| `30m` | 30 dní | Krátkodobé pozice |
| `1h` | 180 dní | Swingtrading (doporučeno) |
| `4h` | 180 dní | Střednědobé pozice |
| `1d` | 6 měsíců | Poziční obchodování (výchozí) |

**Dostupné názvy assetů:**
```
Gold, Silver, Oil, Brent_Oil, USDIDX, SP500, MSCIWorld, Nasdaq100,
Bitcoin, MSFT, HIMS, Nokia, Ericsson, GOOGL, Apple, Tesla, Netflix,
ORCL, NVDA, AMD, Spotify, Coca-Cola, CocaColaCCH, AgnicoEagle,
AEM-CFD, NewmontMining, NovoNordisk, Moneta, KomBanka
```

### Režim 3 – Hodinová analýza všech assetů
Stáhne intraday data pro všechny assety a uloží PNG tabulku signálů. Trvá 1–2 minuty.

```bash
python trading_backtest.py --signals-hourly
python trading_backtest.py --signals-hourly --interval 4h
python trading_backtest.py --signals-hourly --interval 15m
```

**Výstupní soubor:**
```
signals_1h.png    – tabulka signálů z hodinových dat
signals_4h.png    – tabulka signálů ze 4hodinových dat
```

**Jak číst tabulku signálů:**
- **BUY** (zelená) – alespoň 3 z 5 indikátorů souhlasí s nákupem
- **SELL** (červená) – alespoň 3 z 5 indikátorů souhlasí s prodejem
- **NEU** (žlutá) – nejednoznačný signál, lepší počkat
- **`✔`** – indikátor souhlasí s BUY, **`x`** – nesouhlasí

### Jak vybrat nejlepší asset k nákupu
1. Spusť `--signals-hourly` pro přehled všech assetů
2. Filtruj na **BUY signál** (zelená)
3. Preferuj assety kde:
   - RSI je mezi 35–50 (prostor k růstu, ale není přeprodáno)
   - MACD histogram ▲ roste (potvrzení momentu)
   - BB% < 0.4 (cena blízko dolního pásma = levnější vstup)
4. Spusť `--analyze <Asset> --interval 1h` pro detailní analýzu kandidáta
5. Zkontroluj Stop-Loss vzdálenost – ideálně do 4 % od aktuální ceny

### Jak změnit časové období backtestу
```python
START_DATE = "2021-01-01"    # změň na požadované datum
```
Doporučená období:
- `"2021-01-01"` – současné nastavení, ~5 let
- `"2019-01-01"` – zachytí COVID crash 2020
- `"2018-01-01"` – zachytí i předcovidový bull market

## 7. Analýza rychlosti a objemu 
Sekce **SPEED & VOLUME** se zobrazuje na konci výstupu příkazu `--analyze`. Šest metrik doplňuje klasické indikátory (MA, RSI, BB, MACD, ATR) o pohled na **sílu a přesvědčení** aktuálního pohybu ceny. Zatímco hlavní indikátory říkají *kam* cena míří, metriky rychlosti a objemu říkají *jak přesvědčivě*.
 
**Klíčová myšlenka:** BUY signál 3/5 s expanding ATR + bullish OBV + nadprůměrným objemem je výrazně spolehlivější než stejný signál s contracting ATR + OBV divergencí + nízkým objemem. 
 
### 7.1 Rate of Change (ROC)
**Co měří:** Celkový procentuální pohyb ceny za posledních 10 svíček.
 
**Vzorec:**
```
ROC = (cena_nyní - cena_před_10_svíčkami) / cena_před_10_svíčkami × 100
```
 
**Jak hodnotit:**
| Hodnota | Interpretace |
|---------|-------------|
| ROC > +5 % | Přehřátí – cena se rychle vzdálila od průměru, riziko korekce |
| ROC +1 % až +5 % | Zdravý pohyb nahoru |
| ROC -1 % až +1 % | Konsolidace – čekej na průlom |
| ROC -1 % až -5 % | Zdravý pohyb dolů |
| ROC < -5 % | Potenciální oversold – možný odraz |
 
**Důležitost:** (3/5) střední. ROC je nejužitečnější v kombinaci s ATR trendem – rychlý pohyb (vysoké ROC) s expanding ATR je přesvědčivý, rychlý pohyb s contracting ATR může být falešný.
 
**Pozor na timeframe:** ROC +5 % za 10 hodinových svíček (10h) je úplně jiná situace než ROC +5 % za 10 denních svíček (2 týdny). Vždy interpretuj v kontextu zvoleného intervalu.
 
### 7.2 ATR trend
**Co měří:** Zda průměrná velikost svíčky (volatilita) roste nebo klesá oproti hodnotě před 10 svíčkami.
 
**Vzorec:**
```
ATR_změna = (ATR_nyní - ATR_před_10) / ATR_před_10 × 100
```
 
**Jak hodnotit:**
| Hodnota | Stav | Interpretace |
|---------|------|-------------|
| > +15 % | ▲ EXPANDING | Volatilita roste – pohyb má energii, momentum se buduje |
| -15 % až +15 % | → STABLE | Normální podmínky |
| < -15 % | ▼ CONTRACTING | Volatilita klesá – pohyb ztrácí sílu, možný konec trendu |
 
**Důležitost:** (5/5) velmi vysoká. Toto je pravděpodobně nejdůležitější ze všech šesti metrik. Profesionální obchodníci sledují ATR expansion jako potvrzení vstupu – signál s expanding ATR je výrazně spolehlivější.
 
**Praktické použití:**
- Expanding ATR při BUY signálu → vstup má smysl, pohyb má energii
- Contracting ATR při BUY signálu → buď opatrný, pohyb možná končí
- Contracting ATR obecně → typický pro konsolidaci před průlomem oběma směry
 
### 7.3 Candle body size 
**Co měří:** Velikost těla aktuální svíčky (rozdíl Open–Close) oproti 20svíčkovému průměru.
 
**Vzorec:**
```
body_ratio = |Close_nyní - Open_nyní| / průměr(|Close - Open|, posledních 20 svíček)
```

**Jak hodnotit:**
| Poměr | Interpretace |
|-------|-------------|
| > 1.5x | Velká svíčka – silné přesvědčení, trh se rozhodl |
| 0.7–1.5x | Normální svíčka |
| < 0.5x | Malá svíčka / doji – nerozhodnost, nespoléhej na signál |
 
**Důležitost:** (4/5) vysoká. Malá svíčka (doji) při BUY signálu znamená že trh váhá – vstup v takovém momentu je riskantní. Velká bullish svíčka při BUY signálu naopak potvrzuje přesvědčení kupujících.
 
**Doji svíčka** (tělo < 0.3x průměru) je klasický signál nerozhodnosti a často předchází obratu nebo průlomu – trh čeká na katalyzátor.
 
### 7.4 Volume
**Co měří:** Objem aktuální svíčky oproti 20svíčkovému průměru.
 
**Vzorec:**
```
volume_ratio = objem_nyní / průměr(objem, posledních 20 svíček)
```
 
**Jak hodnotit:**
| Poměr | Stav | Interpretace |
|-------|------|-------------|
| > 2.0x | SPIKE | Velký hráč vstoupil – velmi silný signál |
| 1.3–2.0x | ABOVE AVERAGE | Nadprůměrný zájem, potvrzuje pohyb |
| 0.7–1.3x | AVERAGE | Normální aktivita |
| < 0.7x | BELOW AVERAGE | Slabý zájem, buď opatrný |
 
**Důležitost:** (5/5) velmi vysoká – ale **jen pro akcie**. Pro futures (Gold GC=F, Oil CL=F) a forex je objem z Yahoo Finance agregát pouze z jedné burzy a není reprezentativní. Pro akcie (NVDA, Spotify, Apple, MSFT) je objem velmi cenná informace.
 
**Praktické použití:**
- Cena roste + volume spike → skutečný zájem kupujících, vstup potvrzený
- Cena roste + nízký objem → pohyb bez přesvědčení, možná manipulace nebo jen absence prodávajících
 
### 7.5 Volume trend
**Co měří:** Zda objem v posledních 5 svíčkách systematicky roste nebo klesá (lineární regrese).
 
**Jak hodnotit:**
| Trend | Interpretace |
|-------|-------------|
| ▲ Objem roste + cena roste | Zdravý bullish trend – kupující přibývají |
| ▲ Objem roste + cena klesá | Silný prodejní tlak – vyhni se longu |
| ▼ Objem klesá + cena roste | Pozor – trend ztrácí podporu, možný obrat |
| ▼ Objem klesá + cena klesá | Prodejci ztrácejí zájem – možný odraz |
 
**Důležitost:** (3/5) střední. Doplňuje aktuální volume – nestačí vědět že dnešní svíčka má vysoký objem, důležité je zda objem roste systematicky.
 
### 7.6 OBV divergence (On-Balance Volume)
**Co měří:** On-Balance Volume je kumulativní indikátor který přičítá objem při rostoucích svíčkách a odečítá při klesajících. Divergence mezi směrem ceny a OBV je jeden z nejspolehlivějších signálů nadcházejícího obratu.
 
**Vzorec:**
```
OBV = Σ (objem × sign(změna_ceny))
OBV_směr = OBV_nyní - OBV_před_5_svíčkami
```
 
**Jak hodnotit:**
| Cena | OBV | Stav | Interpretace |
|------|-----|------|-------------|
| ▲ roste | ▲ roste | BULLISH potvrzení | Pohyb je zdravý, objem potvrzuje |
| ▼ klesá | ▼ klesá | BEARISH potvrzení | Pokles je opodstatněný |
| ▲ roste | ▼ klesá | ⚠ BEARISH DIVERGENCE | Nebezpečí! Cena roste ale "chytří peníze" prodávají → pravděpodobný obrat dolů |
| ▼ klesá | ▲ roste | ⚠ BULLISH DIVERGENCE | Příležitost! Cena klesá ale "chytří peníze" nakupují → pravděpodobný obrat nahoru |
 
**Důležitost:** (5/5) nejvyšší ze všech šesti metrik pro predikci obratu. OBV divergence používají profesionální obchodníci jako jeden z primárních signálů. Myšlenka je jednoduchá: institucionální investoři ("chytří peníze") nedokážou nakoupit velké pozice aniž by to bylo vidět na objemu – OBV toto odhaluje dříve než se projeví na ceně.
 
**Příklad:**
- NVDA cena roste 3 dny, ale OBV klesá → instituce prodávají do síly retailových kupujících → pravděpodobný obrat
- Spotify cena klesá 5 dní, ale OBV roste → instituce nakupují při každém poklesu → pravděpodobný odraz
 
### 7.7 Jak kombinovat všech 6 metrik
Metriky rychlosti a objemu **nenahrazují** hlavní signály (MA, RSI, BB, MACD, ATR) ale **filtrují** je. Používej je jako druhý krok po získání BUY/SELL signálu.
 
**Ideální vstup do BUY pozice:**
```
Hlavní signál:  BUY ≥ 3/5
─────────────────────────────────────────
ROC:            mírně kladný (ne přehřátý, ideálně +1 až +3 %)
ATR trend:      ▲ EXPANDING – pohyb má energii
Candle body:    ≥ 1.0x průměru – přesvědčení
Volume:         ≥ 1.3x průměru – potvrzení zájmu
Volume trend:   ▲ rostoucí posledních 5 svíček
OBV:            ▲ BULLISH nebo BULLISH DIVERGENCE
```
 
**Signál k vyhnutí se vstupu i přes BUY 3/5:**
```
ATR trend:   ▼ CONTRACTING  → pohyb ztrácí sílu
Volume:      < 0.7x průměru → slabý zájem
OBV:         ⚠ BEARISH DIVERGENCE → instituce prodávají
Candle body: < 0.5x průměru → doji, nerozhodnost
```
 
**Praktické pořadí důležitosti pro rozhodnutí:**
1. OBV divergence – nejdůležitější, předchází ostatním
2. ATR trend – potvrzuje nebo vyvrací energii pohybu
3. Volume – potvrzuje zájem trhu
4. Candle body – potvrzuje přesvědčení aktuální svíčky
5. Volume trend – systémový vs jednorázový zájem
6. ROC – kontext rychlosti, doplňkový

## 8. Monte Carlo predikce
### Co je Monte Carlo simulace

Monte Carlo je statistická metoda která místo jedné předpovědi generuje **tisíce možných scénářů** vývoje ceny a zobrazuje jejich pravděpodobnostní rozložení. Pojmenována podle kasina v Monte Carlu kvůli využití náhodnosti.

### Jak to funguje ve skriptu
Skript používá **jinou metodu simulace pro každý profil assetu** – každý typ trhu má jiné chování a zaslouží si model který to respektuje:

```
1. Vezme denní výnosy posledních 90 dní  →  zjistí volatilitu a parametry
2. Vybere model dle profilu assetu       →  viz tabulka níže
3. Vygeneruje 1000 náhodných cest        →  každý den = krok dle zvoleného modelu
4. Spočítá percentily ze všech cest      →  10%, 25%, 50%, 75%, 90%
5. Vykreslí vějíř pravděpodobnosti       →  30 obchodních dní dopředu
```

### Metody simulace dle profilu
| Profil | Model | Barva | Popis |
|--------|-------|-------|-------|
| `DEFENSIVE` | **Random Walk** | modrá | Prostý GBM – každý den nezávislý náhodný šok. Vhodné pro ETF kde efektivní trh funguje nejlépe. |
| `TECH` | **Random Walk + Earnings** | fialová | Random walk doplněný o náhodné earnings skoky (±8 % průměrně, ~48% šance v 30denním okně). Zachycuje čtvrtletní volatilitu tech akcií. |
| `COMMODITY` | **GBM + Mean Reversion** | oranžová | Přidává přitažlivost k dlouhodobému průměru (1 rok). Komodity mají tendenci vracet se k rovnovážné ceně – zlato k long-term průměru, ropa k výrobním nákladům. |
| `CRYPTO` | **GARCH(1,1)** | červená | Volatilita závisí na předchozí volatilitě a šocích (α=0.15, β=0.80). Zachycuje volatility clustering – klidná období střídají bouřlivé fáze typické pro Bitcoin. |
| `FOREX_IDX` | **Ornstein-Uhlenbeck** | zelená | Silná mean reversion (θ=0.12) – kurzy gravitují k dlouhodobé rovnovážné hodnotě. Mnohem silnější přitažlivost než u komodit. |

## 9. Prophet predikce
Prophet (od Meta/Facebook) je statistický model navržený pro časové řady s trendem a sezónností. Na rozdíl od Monte Carlo, který simuluje náhodný pohyb, Prophet **fituje skutečné vzory** v historických datech.
 
### Jak Prophet funguje
Prophet rozkládá časovou řadu na tři složky: 
```
cena(t) = trend(t) + sezónnost(t) + šum(t)
```
 
**Trend** – automaticky detekuje body obratu (changepoints) kde se trend zlomil. Pokud Gold rostl 6 měsíců a pak se otočil, Prophet to zachytí.
 
**Sezónnost** – týdenní vzory (akcie pondělí vs pátek), roční vzory (zlato v Q4). Fituje se Fourierovou řadou.
 
**Šum** – zbytek po odečtení trendu a sezónnosti.
 
### Anchor na poslední cenu
Prophet interně predikuje od svého fitovaného trendu, který může být výše nebo níže než poslední reálná cena. Skript proto aplikuje **anchor shift** – všechny předpovězené hodnoty jsou posunuty tak aby `yhat[0]` začínal přesně od poslední known close. Tím se zachová tvar a směr křivky ale predikce plynule navazuje na graf.
 
### Vizuální výstup
V grafu `plot_asset` a `--analyze` se Prophet zobrazuje oranžovou barvou vedle modrého Monte Carlo:
```
Oranžová plná linie    → yhat (střední predikce)
Oranžový stínovaný fan → 80% konfideční interval
Oranžová přerušovaná   → čistá trendová komponenta
```
 
### Kdy je Prophet užitečný
| Situace | Doporučení |
|---------|-----------|
| Jasný uptrendový nebo downtrendový asset | Prophet přesněji extrapoluje trend |
| Sezónní asset (Gold Q4, energie zima) | Prophet zachytí sezónní vzor |
| Velmi volatilní asset (Bitcoin, NVDA) | Monte Carlo je spolehlivější |
| Intraday interval (15m, 1h) | Obě metody jsou srovnatelné |

## 10. Streamlit Dashboard
Dashboard poskytuje interaktivní webové rozhraní pro všechny funkce skriptu bez nutnosti pracovat s terminálem.
 
### Instalace
```bash
pip install streamlit plotly 
```
 
### Spuštění 
Oba soubory musí být ve stejné složce:
 
```
Trading_algorithm/
├── trading_backtest_script.py
└── dashboard.py
```
 
```bash
streamlit run dashboard.py
```

Dashboard se automaticky otevře v prohlížeči na `http://localhost:8501`.
 
### Stránky dashboardu
| Stránka | Obsah |
|---------|-------|
| **Signal Overview** | Tabulka signálů všech assetů s BUY zónou, Stop-Loss, Take Profit, SELL target. Grafy distribuce signálů podle profilu. |
| **Asset Detail** | Interaktivní Plotly graf (cena, EMA, BB, RSI, MACD, ATR), indikátorová tabulka, Speed & Volume metriky, Volume Profile, breakout analýza, Monte Carlo + Prophet forecast. |
| **Order Levels** | Tabulka doporučených příkazů (Buy Limit, Stop-Loss, TP1, TP2, Risk USD) pro všechna aktiva. Vizuální cenové hladiny pro vybrané aktivum. |
| **Backtest Summary** | Plný historický backtest seřazený best→worst. Tabulka s Return, B&H, Alpha, Sharpe, Win Rate, Max DD. Grafy Return vs B&H, Sharpe, Win Rate. |
| **Comparison Charts** | Equity křivky více assetů najednou. 4-panelový srovnávací graf. Scatter Return vs Sharpe. |
 
### Nastavení v sidebaru
| Nastavení | Popis | Výchozí |
|-----------|-------|---------|
| **Interval** | Časový rámec pro signály (1h, 4h, 1d, 15m, 30m) | 1h |
| **Initial capital** | Kapitál na asset pro výpočet rizika a backtestů | $10,000 |
| **Start date** | Začátek backtestového období | 2021-01-01 |
| **Signal filter** | Zobrazit jen BUY / SELL / NEU | vše |
| **Profile filter** | Filtr podle profilu (TECH, COMMODITY, …) | vše |
 
### Refresh dat
Kliknutím na **Refresh signals** v sidebaru se vymaže cache a stáhnou nová data. Signály jsou cachované 30 minut, backtesty 2 hodiny.
 
### Poznámky k výkonu 
Plný backtest (Backtest Summary stránka) stahuje data pro všechna aktiva a může trvat 3–5 minut při prvním spuštění. Výsledky jsou cachované – opětovné kliknutí je okamžité dokud nezmáčkneš Refresh nebo nezměníš datum/kapitál.


## 11. Disclaimer
> **Upozornění:** Tento skript slouží výhradně k **vzdělávacím a analytickým účelům**.
> Backtest na historických datech **nezaručuje budoucí výsledky**.
> Výkonnost v minulosti ≠ výkonnost v budoucnosti.
> Skript **nepředstavuje investiční poradenství**.
> Před reálným obchodováním konzultujte licencovaného finančního poradce.
