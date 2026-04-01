![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![Data: yfinance](https://img.shields.io/badge/Data-yfinance-green.svg)

# Multi-Asset Trading Algorithm & Dashboard
Komplexní systém pro automatizovaný backtesting, real-time analýzu a predikci tržních trendů (Monte Carlo & Facebook Prophet) pro komodity, akcie a forex.

## Rychlý start
1. **Instalace:** `pip install -r requirements.txt`
2. **Spuštění Dashboardu:** `streamlit run dashboard.py`
3. **Analýza v terminálu:** `python trading_backtest.py --analyze Gold --interval 1h`

## Obsah
1. [Přehled systému](#1-přehled-systému)
2. [Konfigurace a Profily Assetů](#2-konfigurace-a-profily-assetů)
3. [Technické indikátory a Strategie](#3-technické-indikátory-a-strategie)
4. [Kombinovaná obchodní logika](#4-kombinovaná-obchodní-logika)
5. [Backtesting a Metriky výkonnosti](#5-backtesting-a-metriky-výkonnosti)
6. [Analýza rychlosti a objemu (Speed & Volume)](#6-analýza-rychlosti-a-objemu)
7. [Predikční modely (Monte Carlo & Prophet)](#7-predikční-modely)
8. [Návod k použití a režimy](#8-návod-k-použití-a-režimy)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [Disclaimer](#10-disclaimer)

## 1. Přehled systému
Skript provádí historický **backtest kombinované technické strategie** na 15+ assetech. Systém automaticky stahuje data, počítá indikátory a generuje signály na základě hlasování indikátorů (konsenzus ≥ 3 z 5).

## 2. Konfigurace a Profily Assetů
Každé aktivum má svůj specifický charakter. Algoritmus používá **dynamické profily**:
* **COMMODITY:** Širší pásma, pomalejší průměry (Zlato, Ropa).
* **TECH:** Agresivní nastavení pro vysokou volatilitu (NVDA, AMD).
* **DEFENSIVE:** Konzervativní stop-lossy (Coca-Cola, Moneta).
* **FOREX_IDX:** Úzká pásma pro pomalé pohyby (USD Index).

## 3. Technické indikátory
Strategie kombinuje 5 klíčových pilířů:
* **EMA Crossover (20/50):** Detekce trendu.
* **RSI (14):** Síla hybnosti (Momentum).
* **Bollinger Bands:** Statistická odchylka ceny.
* **MACD:** Konvergence a divergence trendu.
* **ATR:** Dynamické řízení rizika (Stop-Loss).

## 4. Kombinovaná obchodní logika
**Žádný indikátor neobchoduje sám.** * **BUY signál:** Vyžaduje skóre ≥ 3 z 5 (např. EMA roste + RSI < 50 + MACD Bullish).
* **Risk Management:** Automatický Stop-Loss nastaven na `2.0 * ATR`.

## 5. Backtesting a Metriky výkonnosti
Systém počítá pokročilé metriky pro objektivní zhodnocení:
* **Alpha:** Nadvýnos nad strategií "Kup a drž" (Buy & Hold).
* **Sharpe Ratio:** Výnos očištěný o riziko.
* **Max Drawdown:** Největší historický pokles kapitálu.
* **Profit Factor:** Poměr hrubých zisků k hrubým ztrátám.

## 6. Analýza rychlosti a objemu (Speed & Volume)
Tato sekce doplňuje signály o pohled na "přesvědčení" trhu:
* **ROC (Rate of Change):** Rychlost pohybu.
* **OBV Divergence:** Klíčový indikátor pro odhalení nákupů institucí.
* **Candle Body Ratio:** Síla aktuální svíčky.

## 7. Predikční modely
Systém nabízí dva pohledy do budoucnosti:
1.  **Monte Carlo (1000 simulací):** Pravděpodobnostní vějíř (Random Walk, GARCH pro krypto, Mean Reversion pro komodity).
2.  **Facebook Prophet:** Statistický model Meta, který analyzuje sezónnost a trendy.

## 8. Návod k použití a režimy
Skript podporuje tři hlavní režimy:
* `--analyze [Asset]`: Detailní technický rozbor jednoho aktiva.
* `--signals-hourly`: Rychlý přehled signálů napříč trhem (1h/4h intervaly).
* Výchozí běh: Kompletní historický backtest celého portfolia.

## 9. Streamlit Dashboard
Interaktivní webové rozhraní zahrnuje:
* **Signal Overview:** Přehledná tabulka s nákupními zónami.
* **Asset Detail:** Interaktivní grafy Plotly a Volume profily.
* **Backtest Summary:** Vizualizace equity křivek a srovnání assetů.

## 10. Disclaimer
Tento software je určen pouze pro vzdělávací účely. Obchodování na finančních trzích zahrnuje vysoké riziko ztráty. Minulé výsledky nejsou zárukou budoucích výnosů.