# Architektura aplikacji

```
config/default_config.py     parametry: zakresy, zbiory rozmyte, tabela reguł, FOU, symulacja
        │
core/membership_type2.py     IT2TrapezoidMF: UMF, LMF, FOU
core/variables.py            LinguisticVariable + budowniczowie (temperatura, wilgotność, wyjście)
core/rule_base.py            Rule, RULES (20 reguł) – jedyne źródło prawdy
core/inference.py            IT2MamdaniEngine: fuzzyfikacja, siły odpalenia [f, F], agregacja, centroidy
core/type_reduction.py       algorytm Karnika–Mendela, center-of-sets, centroid zbioru IT2
core/controller.py           IT2FuzzyController → ControllerResult (diagnostyka)
core/room_simulation.py      RoomParams, room_step, simulate()
core/metrics.py              MAE, RMSE, czasy, overshoot, energy proxy
core/analysis.py             krzywa odpowiedzi, powierzchnia sterowania, wpływ FOU (dane z silnika)
classical/on_off_controller.py   regulator ON/OFF z histerezą
visualization/*.py           wykresy Matplotlib (zwracają Figure, nie znają Streamlita)
app.py                       interfejs Streamlit – tylko wywołuje silnik i prezentuje wyniki
```

## Odpowiedzialność modułów

| Moduł | Odpowiedzialność | Zależy od |
|---|---|---|
| `config` | stałe i parametry domyślne | – |
| `membership_type2` | matematyka UMF/LMF/FOU, walidacja parametrów | numpy |
| `variables` | zbiory w zmiennej, wektorowe obliczanie przynależności, obcinanie wejść | config, membership |
| `rule_base` | definicja reguł i widok tabelaryczny | config |
| `inference` | $[f_l,F_l]$ (min), agregacja (max), centroidy konsekwencji | variables, rule_base, type_reduction |
| `type_reduction` | KM (`km_endpoint`), `type_reduce_center_of_sets`, `it2_centroid` | numpy |
| `controller` | cały tor: $e$ → fuzzyfikacja → reguły → KM → crisp → status | inference, type_reduction |
| `room_simulation` | model termiczny i pętla zamknięta dla dowolnego regulatora z metodą `control()` | config |
| `metrics` | metryki z danych symulacji | numpy, pandas |
| `visualization` | prezentacja wyników silnika | matplotlib |
| `app.py` | stan sesji, układ UI, cache | wszystko powyżej |

## Przepływ danych

```
UI (T_room, T_set, h)
   └─► IT2FuzzyController.compute()
         e = T_room − T_set ─ clip ─► LinguisticVariable.memberships()   [LMF, UMF]
         ─► IT2MamdaniEngine.fire()        [f_l, F_l] dla 20 reguł (min)
         ─► IT2MamdaniEngine.aggregate()   zagregowany zbiór (do wykresu)
         ─► type_reduce_center_of_sets()   KM → [y_left, y_right]
         ─► crisp = (y_left + y_right)/2 → status HEATING/NEUTRAL/COOLING
   ◄── ControllerResult (membershipy, reguły, [f, F], agregacja, KM, crisp, stan)
```

## Połączenie UI z silnikiem

* `app.py` trzyma w `st.session_state` tylko wartości (T_room, wilgotność, T_set i ustawienia). Przyciski `+`/`−`
  używają callbacków `adjust()` z obcinaniem do zakresów (0.5 °C, 1 %, 0.5 °C).
* Każde przeładowanie skryptu wywołuje `IT2FuzzyController.compute()` dla aktualnych wartości, więc zmiana któregokolwiek
  wejścia natychmiast zmienia błąd, przynależności, reguły, wynik, status i wykresy. UI nie zawiera żadnej logiki rozmytej ani wartości „na sztywno”.
* Ciężkie, niezależne od stanu bieżącego obliczenia są cache'owane (`st.cache_data`): powierzchnia sterowania,
  krzywa odpowiedzi, wpływ FOU i symulacje (klucz: parametry + współczynnik FOU + histereza).
* Zakładki działają w trybie leniwym (`st.tabs(..., on_change="rerun")` i `tab.open`): wykonywany jest kod tylko aktywnej
  zakładki, dzięki czemu kliknięcie `+`/`−` odświeża się w ułamku sekundy. Dla starszych wersji Streamlit jest fallback
  do zwykłych zakładek. Wartości widgetów z nieaktywnych zakładek są zachowywane (`session_state[k] = session_state[k]`).

## Działanie symulacji

`simulate(controller, params)` wykonuje pętlę $T_k \to u_k=\text{controller.control}(T_k,T_{set},h) \to T_{k+1}$
(całkowanie wykładnicze). Ten sam model obsługuje IT2FLS (`IT2FuzzyController.control`) i ON/OFF (`OnOffController.control`),
więc porównanie jest uczciwe: ta sama fizyka, to samo $T_0$, $T_{out}$, $\Delta t$. Wynik to `DataFrame`
(`time_min, t_room, t_set, error, u, heating, cooling`), z którego `core.metrics.compute_metrics` liczy metryki.

## Testy

`tests/` zawiera testy jednostkowe każdego modułu oraz testy interfejsu (`streamlit.testing.v1.AppTest`):
kliknięcia `+`/`−`, granice zakresów, aktualizacja wyniku po zmianie wejść, trwałość ustawień i renderowanie wszystkich zakładek.
