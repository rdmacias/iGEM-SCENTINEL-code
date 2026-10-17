# Cómo trata la herramienta 1 los datos que faltan

## 1. El problema

La tabla de Ki tiene 621 compuestos × 437 proteínas, es decir, 271.377 pares posibles.
**Solo 7.852 están medidos en el laboratorio: un 2,9 %.** Las predicciones Tanimoto
rellenan otros ~100.000 pares, cada uno con su confianza (0–1). El resto está vacío.

Para saber si una mezcla de interferentes puede imitar al compuesto objetivo (MeSA), el
programa necesita un número en cada casilla del panel. Cuando no lo hay, tiene que
suponer uno, y **cualquier suposición puede hacer que un panel parezca mejor de lo que es.**

El programa no puede inventar los datos que faltan. Lo que sí hace es:
1. elegir con cuidado qué datos usa,
2. evitar las proteínas de las que se sabe poco,
3. **decir de forma explícita y con números en qué supuestos se apoya el resultado.**

## 2. Las capas, en orden

### Capa 1 – Qué datos se usan (`--source`, `--min-confidence`)

| Modo | Datos del objetivo | Datos de los interferentes | Casilla vacía |
|---|---|---|---|
| `experimental` | solo medidos | solo medidos | "no se une" (señal 0) |
| `predicted` | medidos + predicción central (`imputed_pred`) | medidos + **predicción conservadora** (`imputed_lower`: se suponen tan fuertes como el límite inferior) | "no se une" (señal 0) |

- `--min-confidence 0.3` descarta las predicciones con confianza < 0,3. **Ojo: una
  predicción descartada pasa a contar como "no se une"**, que también es una suposición.
- Los valores censurados del ensayo (">30 µM") se usan como 1,1 × 30 = 33 µM: se supone que
  se unen un poco, lo que es prudente para la herramienta 1.

**Por qué "no se une" para lo que falta:** el cálculo necesita un número, y 0 es la única
opción neutra. Pero es **optimista**: un interferente que no conocemos no puede imitar
nada, así que el panel parece más específico de lo que es. Las capas 2 y 4 existen para
corregir ese optimismo.

### Capa 2 – Antes de buscar: proteínas ciegas y filtro c2

**Proteínas ciegas fuera.** Si una proteína solo tiene valores ">X" para el objetivo y
todos los interferentes, no se une a ninguno de ellos y un sensor con ella no daría
señal. Aun así, la regla ">X" → 1/(1,1·X) le daría una señal pequeña e igual para todos, y
el cálculo la usaría como si aportara información. Por eso se descarta. Con MeSA y los 5
interferentes, 24 de las 58 proteínas con c2 ≥ 0,5 eran ciegas.

**Filtro c2 (`--min-c2`, por defecto 0,5).**

Para cada proteína, c2 mide cuánto se sabe de los interferentes en ella:
medido = 1, predicho = su confianza, vacío o descartado = 0; se suman y se divide por el
número de interferentes. Solo las proteínas con c2 ≥ 0,5 pueden entrar en el panel.

Limitaciones: la proteína inicial (`--start-with`) entra siempre, aunque tenga c2 bajo (el
programa avisa). Y c2 = 0,6 todavía permite que falten 2 de 5 interferentes.

### Capa 3 – La búsqueda (NNLS + greedy)

El programa va añadiendo proteínas. Con cada panel calcula la mezcla de interferentes que
más se parece al objetivo (NNLS) y la distancia que queda (el residual). Se para cuando
el residual supera el ruido del sensor (`--noise`, por defecto 0,05).

**Cómo elige la siguiente proteína (`--priority`)** depende del modo:

| Modo | Prioridad | Regla |
|---|---|---|
| 1 – solo medidos | `measured` | Entre las proteínas que suben el residual al menos `--min-gain`, elige la de mayor **ganancia × c2²**. Prefiere una proteína bien medida aunque suba algo menos el residual. Objetivo: un panel en el que se pueda confiar. |
| 2 – con predicciones | `score` | Elige la que más sube el residual, **contando las predicciones**. Puede encontrar paneles que el modo 1 no ve, y las apuestas dicen qué medir para confirmarlos. |

Se puede forzar cualquiera de las dos en cualquier modo con `--priority score` o
`--priority measured`.

**La búsqueda solo ve los números de la capa 1.** Si al linalool le falta el dato en una
proteína, para la búsqueda ese linalool no existe en ella. Por eso hace falta la capa 4.

### Capa 3b – Cobertura: ningún interferente invisible (`--cover`, por defecto 1)

Cuando termina la búsqueda, el programa comprueba que **cada interferente se una de verdad
a al menos una proteína del panel** (Ki medido que no sea ">X", o una predicción usada). Un
">X" no sirve: la proteína no se une, así que tampoco notaría el interferente. Si alguno
no cumple, añade la proteína que sí se une a él y deja el residual más alto. En la tabla
aparece con `cover` y `<- binds L-Linalool`.

Por qué: un interferente que no se conoce en ninguna proteína del panel es invisible.
Si resulta que se une a la proteína principal, puede imitar al objetivo "gratis",
porque basta con subir su concentración y no se nota en ninguna otra proteína. Una sola
proteína donde esté medido lo impide. Añadir una proteína nunca baja el residual.

Si ninguna proteína candidata (c2 ≥ 0,5) se une a él, se busca entre todas las que tienen
datos del objetivo. Si tampoco hay ninguna, el programa avisa de que ese interferente
sigue sin comprobar. `--cover 0` desactiva este paso; `--cover 2` exige dos proteínas.

### Capa 4 – Después: ¿en qué apuesta este panel?

Cuando el panel pasa, el programa revisa **cada casilla que no está medida** (y cada
valor medido que esté justo en el límite) **de una en una**:

1. Prueba todos los Ki posibles para esa casilla, de 0,1 µM a 10 mM.
2. Mira para qué Ki el residual baja del ruido, es decir, el panel deja de servir.
3. Compara ese rango con la predicción Tanimoto (o con 30 µM si no hay predicción).

El resultado se escribe como **apuestas**: frases sobre lo que tiene que ser verdad para
que el panel siga funcionando, cada una con el dato en que se basa.

## 3. Cómo leer el resultado

```
VERDICT: WEAK  -  panel DcitOBP3 + CsupOBP17 + GmolCSP8 + AlinOBP6
Passes on the data used (residual 0.0565 vs noise 0.05); 2 bet(s) go AGAINST the Tanimoto prediction.
If every not-measured pair took its Tanimoto prediction at once: residual 0.0000 -> FAILS

WHAT THIS PANEL BETS ON (6 unverified assumptions; most fragile first):
  1. hexanal binds DcitOBP3 weakly or not at all (Ki > 15.8 uM)  AGAINST prediction
     data: not measured; prediction Ki 3.77 uM, conf 0.47, left out (counted as 'does not bind')
  2. L-Linalool does NOT bind DcitOBP3 at all  AGAINST prediction
     data: not measured; prediction Ki 3.09 uM, conf 0.59, left out ...
```

**Veredicto**

| Veredicto | Significado |
|---|---|
| `SOLID` | Ningún dato sin medir ni ningún valor medido puede romperlo de forma creíble. |
| `CONDITIONAL` | Depende de valores sin medir, pero la predicción da margen. Mira cuánto margen hay en "off by". |
| `WEAK` | Al menos una apuesta va **en contra de la predicción**, o el panel falla si todas las predicciones son ciertas a la vez. No es fiable hasta medir. |

**Cada apuesta**

- `Ki > X` → el panel necesita que ese interferente se una **débilmente** (o nada). Si
  se une más fuerte que X µM, lo imita.
- `Ki < X` → el panel necesita que esa unión predicha sea **fuerte**. Si es más débil,
  pierde lo que le permitía distinguir.
- `below X or above Y` → el panel falla si el Ki real cae entre X e Y.
- `does NOT bind at all` → el interferente no da señal en ninguna otra proteína del panel.
  En el modelo lineal la concentración no tiene límite: aunque se una débil, basta con
  poner más. Hay que medirlo **en las otras proteínas**, no solo en esta.
- `AGAINST prediction` → la predicción Tanimoto dice que la apuesta es falsa.
- `lost if off by xN` → cuántas veces tendría que equivocarse la predicción (o el valor
  medido) para perder la apuesta. ×1,1 = basta un error del 10 %; ×10 = muy seguro.
- `Ki ... is accurate (fails if it is 9 % higher)` → un valor **medido** del que depende
  el resultado. Un ensayo suele tener un error del 10–20 %: conviene repetirlo.

Las apuestas que solo se pierden si la predicción falla más de 10 veces se resumen en una
línea. Todas están en `results/tool1_bets_*.csv`. El detalle completo (rango de Ki que
rompe, predicción, residual con la predicción) está en `results/tool1_weak_points_*.csv`.

## 4. Ejemplo: MeSA frente a 5 interferentes

| | Con solo datos medidos | Con todas las predicciones |
|---|---|---|
| Comando | `--source experimental` | `--source predicted --min-confidence 0` |
| Panel | DcitOBP3 + AlinOBP6 + LstiGOBP2 (LstiGOBP2 se une al linalool, Ki 16,5 medido) | DcitOBP3 + PxylGOBP2 + CmedOBP14 |
| Pares por medir | 5 | 11 |
| Residual | 0,053 (umbral 0,05) | 0,575 |
| Veredicto | **WEAK** | **CONDITIONAL** |
| Apuesta principal | el hexanal y el linalool se unen a DcitOBP3 con Ki > 8 y > 25 µM (la predicción dice Ki ≈ 3,8 y 3,1) | el MeSA se une a CmedOBP14 con Ki < 14 µM (predicción 1,4, conf 0,26) |

El punto débil del modo 1 no son las proteínas que elige, sino DcitOBP3, que se pone
como proteína inicial y a la que le faltan 3 de los 5 interferentes. Los dos panels se
apoyan en cosas distintas. Lo que hay que medir lo dice el propio
programa: el hexanal y el linalool en DcitOBP3, y el MeSA en CmedOBP14. Con 3 medidas se
sabe cuál de los dos vale, en vez de las 8–11 de cada lista completa.

## 5. Forma de trabajar recomendada

1. Ejecuta los dos modos (`experimental` y `predicted`) con la misma proteína inicial.
2. Compara los veredictos y las 3–5 primeras apuestas de cada uno.
3. Mide en el laboratorio los pares de esas apuestas (primero los `AGAINST prediction` y
   los de "off by" pequeño).
4. Añade las medidas a `Compound_OBP_binding.csv` y vuelve a ejecutar. Las apuestas
   ganadas desaparecen; si se pierde alguna, el programa elegirá otro panel.
5. Mide el ruido real de vuestro sensor y pásalo con `--noise`: el 0,05 es un valor por defecto.

## 6. Lo que este análisis no hace

- **Cambia las casillas de una en una.** Dos valores desconocidos juntos pueden hacer más
  daño que cada uno por separado. La línea "If every not-measured pair took its Tanimoto
  prediction at once" lo comprueba en parte, con todas las predicciones a la vez.
- **Los ">X" siguen entrando en el cálculo como 1/(1,1·X)** en las proteínas que no son
  ciegas. Para los interferentes es prudente (se supone que se unen un poco); para el
  objetivo es una aproximación.
- **No limita la concentración de los interferentes.** Por eso salen casos de "any Ki".
  En la realidad, un interferente a 400 veces la concentración del MeSA puede ser o no
  realista según el viñedo.
- **Supone respuesta lineal** (concentración muy por debajo del Ki) y que no hay
  interferentes fuera de la lista.
- **El veredicto es tan bueno como la lista de interferentes**: un compuesto que no está
  en la lista no se comprueba.

---

# Herramienta 2 (J de Fisher): cómo trata los datos que faltan

## Qué mide J

Para cada panel, el programa calcula la lectura que daría cada planta:
**lectura = Σ señal(proteína, compuesto) × cantidad del compuesto en la planta**.
Después compara las plantas sanas con las enfermas:

**J = (lectura media enferma − lectura media sana)² / (varianza sana + varianza enferma + ruido del sensor)**

- Numerador: cuánto cambia la lectura con la enfermedad.
- Denominador: cuánto varía de una planta a otra, más el 5 % de ruido del sensor (`--sensor-noise`).
- Con varias proteínas se usa la versión multivariante (distancia de Mahalanobis), que
  además puede restar el ruido que comparten dos proteínas.

## Las dos incertidumbres

1. **La variación entre plantas: J ya la tiene en cuenta** (es el denominador).
2. **Los Ki sin medir (?)**: cuentan como "no se pega" (señal 0), igual que en la herramienta 1.

## Por qué un ? puede inflar J

Un ? esconde un compuesto. Si en realidad se pega a la proteína, su variación entre
plantas se suma al denominador. Si ese compuesto separa peor las clases que lo que la
proteína ya lee (varía mucho, o cambia poco), J baja. Por eso el ? hacía que J pareciera
más alta de lo que es. Solo si el compuesto escondido separa mejor, J subiría.

Ejemplo real (Bianca 2017, 48 h): CsinGOBP1 tiene J = 17 porque solo tiene medido el
E-2-hexenol (ausente en las sanas). Le faltan 10 de los 13 Ki. Si el (E)-2-hexenal, que no
cambia con la enfermedad, se pegara con Ki ≈ 30 µM, J caería a 0.

## Qué hace el programa

Cuando termina la búsqueda, mueve cada Ki sin medir del panel de 0,1 µM a 10 mM, de uno
en uno, y recalcula J. Una casilla "rompe" el panel si puede bajar J por debajo del límite.

### El límite: cuántas plantas clasificaría mal (`--max-error`, por defecto 5 %)

J se traduce a **porcentaje de plantas mal clasificadas** (sanas que el sensor da por
enfermas, o al revés):

**error = Φ(−√(J/2))**, siendo Φ la curva normal.

Esto supone que las lecturas siguen una campana de Gauss, con la misma dispersión en
sanas y enfermas, y que el sensor corta justo en medio de las dos medias. Es una cifra
idealizada: con n = 3 plantas por grupo la dispersión se conoce de forma aproximada.

| J | Errores (de cada 100 plantas) |
|---|---|
| 1 | 24 |
| 2 | 16 |
| 4 | 8 |
| **5,4** | **5 (límite por defecto)** |
| 8 | 2,3 |
| 13,8 | 0,4 |
| 17 | 0,2 |

Con `--max-error 0.05`, el panel tiene que tener J ≥ 5,4. Un Ki sin medir es una apuesta
si algún valor suyo hace bajar J de 5,4, es decir, si con él el sensor fallaría en más del
5 % de las plantas.

Otras opciones: `--weak-frac 0.5` (el límite es la mitad de la J del panel) o
`--target-score X` (J ≥ X). La J y el porcentaje de errores aparecen en cada línea del output.

- **VERDICT**: SOLID / CONDITIONAL / WEAK (o NOT ENOUGH si ni con los datos usados llega al límite).
- **Ki conocidos por proteína**: medidos / predicción usada / sin dato.
- **WHAT THIS PANEL BETS ON**: cada apuesta, el valor al que baja J y el % de errores
  (`J 17.17 -> 0.00 at Ki 31.6 uM (misclassified 0.2 % -> 48.9 %)`), si el compuesto cambia con la enfermedad o solo varía, y qué dice la predicción.
- **J si todas las predicciones fueran ciertas a la vez.**

Todo se guarda en `results/tool2_bets_*.csv`.


## Ejemplo: MeSA + los interferentes de `interferents_example.txt`

Ficheros en `examples/tool2_mesa_interferentes/`. Datos de Bianca 2017, 48 h. El limoneno y
el hexanal no están en ese estudio: en dos de los ficheros se añaden con valores INVENTADOS
(iguales en sanas y enfermas) para ver el efecto de un compuesto ruidoso escondido.

| Caso | Panel | J | Errores | Veredicto |
|---|---|---|---|---|
| Datos reales (MeSA, hexanol, benzaldehído, linalool) | GmolPBP2 + BminOBP9 P99A + AcerOBP11 | 13,4 | 0,5 % | SOLID |
| + limoneno 800 ± 400 inventado, 1 proteína | GmolPBP2 | 11,1 | 0,9 % | CONDITIONAL: si el limoneno (sin medir) se pega con Ki < 15 µM, J baja a 0,5 (30 % de errores) |
| + limoneno 800 ± 400 inventado, 3 proteínas | GmolPBP2 + OcorOBP1 + AcerOBP11 | 13,4 | 0,5 % | SOLID: aunque el limoneno se pegara a las 3, J solo baja a 9,9 |

Por qué el panel de 3 resiste: si el limoneno se pega a las tres proteínas, las tres ven la
misma variación del limoneno, y la J con varias proteínas puede restarla (una sirve de
referencia de ruido). Con una sola proteína no hay nada con qué restar.

El programa comprueba cada Ki sin medir por separado y, además, cada compuesto en todas las
proteínas del panel donde falta a la vez (filas `ALL n:` del CSV de apuestas).
