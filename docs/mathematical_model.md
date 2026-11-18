# Model matematyczny – Fuzzy HVAC Simulator (IT2FLS)

Dokument opisuje dokładnie to, co jest zaimplementowane w kodzie (`config/default_config.py`, `core/*`, `classical/*`).
Wzory są zapisane w składni LaTeX (renderowane przez GitHub).

## 1. Zmienne i konwencje

| Symbol | Znaczenie | Zakres |
|---|---|---|
| $T_{room}$ | temperatura pomieszczenia | 10–40 °C (UI) |
| $T_{set}$ | temperatura zadana | 16–30 °C |
| $e = T_{room} - T_{set}$ | błąd temperatury | ograniczany do $[-10, 10]$ °C |
| $h$ | wilgotność względna | $[0, 100]$ % |
| $u$ | wyjście regulatora | $[-100, +100]$ |

**Konwencja znaku:** $e<0$ – za zimno (ogrzewanie), $e>0$ – za ciepło (chłodzenie); $u<0$ – ogrzewanie,
$u>0$ – chłodzenie, $u=0$ – neutralnie. Wartości poza zakresem są obcinane (`clip`) do dziedziny modelu, a wynik do $[-100,100]$.

## 2. Przedziałowe funkcje przynależności typu 2

Każdy zbiór jest trapezem $(a,b,c,d)$ (trójkąt: $b=c$). Zbiór „ramieniowy” (shoulder) ma $a=b=-\infty$ lub $c=d=+\infty$.

$$
T(x;a,b,c,d)=\max\!\Big(0,\ \min\!\big(1,\ \tfrac{x-a}{b-a},\ \tfrac{d-x}{d-c}\big)\Big)
$$

**Niepewność** opisuje parametr $s\ge 0$ (przesunięcie krawędzi w jednostkach dziedziny):

$$
\overline{\mu}(x)=\max\!\Big(0,\min\!\big(1,\tfrac{x-(a-s)}{b-a},\tfrac{(d+s)-x}{d-c}\big)\Big)\qquad(\text{UMF})
$$
$$
\underline{\mu}(x)=\max\!\Big(0,\min\!\big(1,\tfrac{x-(a+s)}{b-a},\tfrac{(d-s)-x}{d-c}\big)\Big)\qquad(\text{LMF})
$$

UMF ma krawędzie rozsunięte, LMF zsunięte, przy zachowaniu nachylenia zboczy. Każdy składnik minimum w LMF jest
$\le$ odpowiedniemu składnikowi w UMF, więc **$\underline{\mu}(x)\le\overline{\mu}(x)$ z konstrukcji**.
Dla $s=0$ UMF = LMF (system typu 1).

**Footprint of Uncertainty:**
$$
\mathrm{FOU}(x)=\big[\underline{\mu}(x),\ \overline{\mu}(x)\big],\qquad w(x)=\overline{\mu}(x)-\underline{\mu}(x)\ge 0 .
$$

Szerokość FOU zależy od współczynnika $\kappa\in[0,3]$ (domyślnie 1): $s = \kappa\cdot s_{\text{var}}$, gdzie
$s_{\text{temp}}=0.25$ °C, $s_{\text{hum}}=3\,\%$, $s_{\text{out}}=4$ jedn. sterowania. W trójkątnym szczycie LMF ma wysokość
$<1$ (np. $1-s/(b-a)$), więc FOU jest niezerowe także w rdzeniu zbioru.

### Parametry zbiorów (nominalne, $s=0$)

**Błąd temperatury $e$ [°C]**

| Zbiór | $(a,b,c,d)$ |
|---|---|
| Very Cold | $(-\infty,-\infty,-4,-2)$ |
| Cold | $(-4,-2,-2,-0.4)$ |
| Comfort | $(-2,-0.4,0.4,2)$ |
| Warm | $(0.4,2,2,4)$ |
| Very Warm | $(2,4,+\infty,+\infty)$ |

**Wilgotność $h$ [%]**

| Zbiór | $(a,b,c,d)$ |
|---|---|
| Low | $(-\infty,-\infty,25,40)$ |
| Optimal | $(25,40,55,70)$ |
| High | $(55,70,70,85)$ |
| Very High | $(70,85,+\infty,+\infty)$ |

**Wyjście $u$**

| Zbiór | $(a,b,c,d)$ |
|---|---|
| Strong Heating | $(-\infty,-\infty,-90,-65)$ |
| Moderate Heating | $(-90,-65,-65,-30)$ |
| Gentle Heating | $(-65,-30,-30,0)$ |
| Neutral | $(-30,0,0,30)$ |
| Gentle Cooling | $(0,30,30,65)$ |
| Moderate Cooling | $(30,65,65,90)$ |
| Strong Cooling | $(65,90,+\infty,+\infty)$ |

Zbiory temperatury, wilgotności i wyjścia tworzą (nominalnie) podział jedności: na każdym zboczu suma sąsiednich
przynależności wynosi 1, więc nie ma luk w pokryciu.

## 3. Operatory

* T-norm (spójnik AND oraz implikacja): $a\wedge b=\min(a,b)$.
* S-norm (agregacja reguł): $a\vee b=\max(a,b)$.

## 4. Baza reguł (20 reguł)

$R_k$: IF $e$ is $T_k$ AND $h$ is $H_k$ THEN $u$ is $B_k$. Wiersze – temperatura, kolumny – wilgotność:

| $e$ \ $h$ | Low | Optimal | High | Very High |
|---|---|---|---|---|
| Very Cold | Strong Heating | Strong Heating | Strong Heating | Moderate Heating |
| Cold | Moderate Heating | Moderate Heating | Gentle Heating | Gentle Heating |
| Comfort | Neutral | Neutral | Neutral | Gentle Cooling |
| Warm | Gentle Cooling | Moderate Cooling | Moderate Cooling | Strong Cooling |
| Very Warm | Moderate Cooling | Strong Cooling | Strong Cooling | Strong Cooling |

Temperatura jest głównym czynnikiem. Wilgotność jedynie modyfikuje intensywność: wysoka wilgotność wzmacnia chłodzenie
i osłabia ogrzewanie, niska działa odwrotnie. W strefie komfortu wilgotność może wywołać co najwyżej *Gentle Cooling*
(ID reguł: 1–4 Very Cold, 5–8 Cold, 9–12 Comfort, 13–16 Warm, 17–20 Very Warm, w kolejności kolumn tabeli).
Model **nie jest** fizycznym regulatorem wilgotności.

## 5. Wnioskowanie Mamdaniego (interval type-2)

Dla wejść $(e,h)$ i reguły $l$ przedział siły odpalenia $[f_l, F_l]$:

$$
f_l=\min\big(\underline{\mu}_{T_l}(e),\underline{\mu}_{H_l}(h)\big),\qquad
F_l=\min\big(\overline{\mu}_{T_l}(e),\overline{\mu}_{H_l}(h)\big),\qquad f_l\le F_l .
$$

Zagregowany przedziałowy zbiór wyjściowy (wizualizacja):
$$
\overline{\mu}_{agg}(y)=\max_l \min\big(F_l,\overline{\mu}_{B_l}(y)\big),\qquad
\underline{\mu}_{agg}(y)=\max_l \min\big(f_l,\underline{\mu}_{B_l}(y)\big).
$$

## 6. Redukcja typu – Karnik–Mendel (center-of-sets)

**Centroidy konsekwencji.** Dla każdego zbioru wyjściowego $B$ (typu 2) wyznaczamy przedział centroidu $[c_l(B),c_r(B)]$
algorytmem KM na siatce $y_i$ (2001 punktów) z wagami $[\underline{\mu}_B(y_i),\overline{\mu}_B(y_i)]$.
Dla domyślnego FOU: $c_l=[-89.4,-65.7,-35.7,-4.0,27.7,57.6,85.6]$, $c_r=[-85.6,-57.6,-27.7,4.0,35.7,65.7,89.4]$
(kolejność: Strong Heating … Strong Cooling).

**Type reduction.** Dla reguł z $F_l>0$:

$$
y_l=\min_{f_i\in[f_i,F_i]}\frac{\sum_i f_i\,c^{\,l}_i}{\sum_i f_i},\qquad
y_r=\max_{f_i\in[f_i,F_i]}\frac{\sum_i f_i\,c^{\,r}_i}{\sum_i f_i}.
$$

Ekstrema leżą na brzegach przedziałów $f_i\in\{\underline f_i,\overline F_i\}$; algorytm KM znajduje punkt przełączenia.

**Algorytm (jeden koniec, `core/type_reduction.py::km_endpoint`):**

1. Odrzuć elementy z $F_i=0$; posortuj $y_i$ rosnąco: $y_1\le\dots\le y_N$ (wagi razem z nimi).
2. $f_i=(\underline f_i+\overline F_i)/2$; $y'=\sum f_iy_i/\sum f_i$.
3. Znajdź $k\in\{1,\dots,N-1\}$: $y_k\le y'\le y_{k+1}$.
4. Prawy koniec $y_r$: $f_i=\underline f_i$ dla $i\le k$, $\overline F_i$ dla $i>k$.
   Lewy koniec $y_l$: $f_i=\overline F_i$ dla $i\le k$, $\underline f_i$ dla $i>k$. Policz $y''=\sum f_iy_i/\sum f_i$.
5. Jeśli $|y''-y'|\le 10^{-12}\max(1,|y'|)$ – koniec ($y_l$ lub $y_r = y''$, $L$ lub $R=k$); w przeciwnym razie $y'\leftarrow y''$, krok 3.

Zbieżność następuje w co najwyżej $N$ iteracjach (zabezpieczenie: $2N+10$, flaga `converged`). $y_l$ używa lewych końców
centroidów, $y_r$ – prawych. Przypadki brzegowe: wszystkie $F_i=0$ → wyjątek (sterownik zwraca wtedy bezpiecznie $u=0$);
jedna aktywna reguła → $y_l=y_r=y$; niepoprawne przedziały ($\underline f>\overline F$), NaN/inf, wartości ujemne → `ValueError`.
Poprawność jest sprawdzana testem względem przeglądu zupełnego wszystkich $2^N$ wyborów brzegów (`tests/test_type_reduction.py`).

**Przykład** ($T_{room}=20.5$, $T_{set}=22$, $h=60\%$, FOU×1): aktywne reguły R6 (Cold/Optimal) $[0.467,0.844]$,
R7 (Cold/High) $[0.133,0.533]$, R10 (Comfort/Optimal) $[0.156,0.469]$, R11 (Comfort/High) $[0.133,0.469]$.
KM daje $y_l=-48.43$ ($L=1$, 2 iteracje), $y_r=-17.42$ ($R=2$, 3 iteracje), czyli $u=-32.92$ (ogrzewanie).

## 7. Defuzzyfikacja

$$
u=\operatorname{clip}\!\Big(\frac{y_l+y_r}{2},\,-100,\,100\Big).
$$

Ponieważ $u$ jest średnią centroidów zbiorów, wartości skrajne wynoszą ok. $\pm 87$–$89$ (centroid zbioru skrajnego),
a nie dokładnie $\pm100$. Status: $|u|<3$ → NEUTRAL; $u<0$ → HEATING; $u>0$ → COOLING.

## 8. Model pomieszczenia

Czas w minutach, jedna strefa, bezwładność pierwszego rzędu:

$$
\frac{dT}{dt}=\frac{T_{out}-T}{\tau}+P(u),\qquad
P(u)=r_h\,\frac{\max(-u,0)}{100}-r_c\,\frac{\max(u,0)}{100},
$$

gdzie $\tau$ – stała czasowa strat [min], $r_h,r_c$ – maksymalne szybkości ogrzewania/chłodzenia [°C/min].
Przy stałym $u$ w kroku $\Delta t$ rozwiązanie jest dokładne (bezwarunkowo stabilne):

$$
T_{eq}=T_{out}+\tau P(u),\qquad T_{k+1}=T_{eq}+(T_k-T_{eq})\,e^{-\Delta t/\tau}\;[+\,\varepsilon_k].
$$

$\varepsilon_k\sim\mathcal N(0,\sigma^2)$ – opcjonalne zakłócenie z ustalonym ziarnem (domyślnie $\sigma=0$, symulacja deterministyczna).
Domyślnie: $T_{out}=12$ °C, $T_0=17$ °C, $T_{set}=22$ °C, $\tau=300$ min, $r_h=r_c=0.25$ °C/min, $\Delta t=1$ min, 240 min.
Parametry są walidowane względem `SIM_LIMITS`.

**Uwaga:** regulator IT2FLS jest typu proporcjonalnego (brak członu całkującego), więc przy stałych stratach ciepła
ustala się z niewielkim uchybem: $P(u)=(T-T_{out})/\tau$ wymaga $u\ne0$, a $u\ne0$ wymaga $e\ne0$.

## 9. Regulator ON/OFF z histerezą

Stany OFF/HEAT/COOL z pamięcią ($h_{ys}$ – histereza, domyślnie 0.5 °C):

* OFF → HEAT, gdy $T\le T_{set}-h_{ys}$; HEAT → OFF, gdy $T\ge T_{set}$ (wyjście $u=-100$);
* OFF → COOL, gdy $T\ge T_{set}+h_{ys}$; COOL → OFF, gdy $T\le T_{set}$ (wyjście $u=+100$).

Działa w tej samej symulacji pomieszczenia, z tą samą konwencją znaku $u$.

## 10. Metryki

Niech $e_k=T_k-T_{set}$, $N$ – liczba próbek, $\Delta t$ – krok.

* $\mathrm{MAE}=\frac1N\sum|e_k|$, $\ \mathrm{RMSE}=\sqrt{\frac1N\sum e_k^2}$.
* **Time to comfort** – pierwszy czas, dla którego $|e_k|\le b$ ($b$ – półszerokość strefy komfortu, domyślnie 1 °C).
* **Settling time** – czas, od którego $|e_k|\le b$ do końca symulacji (brak → „—”).
* **Overshoot** – maks. przekroczenie $T_{set}$ w kierunku dojścia; **undershoot** – maks. spadek po stronie przeciwnej po pierwszym osiągnięciu $T_{set}$.
* **Control changes** – liczba kroków z $|u_k-u_{k-1}|>5$.
* **Energy proxy** $=\sum_k \frac{|u_k|}{100}\Delta t$ [minuty pełnej mocy]. To wskaźnik intensywności sterowania, **nie** zużycie energii w kWh.
