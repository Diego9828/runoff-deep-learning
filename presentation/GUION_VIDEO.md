# Guion del video explicativo

**Proyecto:** Pronóstico horario de caudal con DMEL (MRS + ensamble dual Informer–LSTM)  
**Duración objetivo:** 17 minutos y 30 segundos  
**Límite:** 18 minutos

> Las indicaciones entre corchetes no se narran. Los tiempos son acumulados y
> sirven como referencia. Las diapositivas del apéndice se reservan para
> preguntas.

---

# 1. Metodología — 0:00 a 6:00

## Slide: Portada

**Tiempo: 0:00–0:10**

“Hola. En este video presentamos nuestra implementación de DMEL, un ensamble
dual para pronosticar caudal, adaptado al pronóstico horario de 508 cuencas.”

[Pasar la portada de sección “Metodología” sin detenerse.]

## Slide: Catorce días para ver dos días adelante

**Tiempo: 0:10–0:50**

“El problema es el siguiente: con las últimas 336 horas, es decir, 14 días, de
caudal y de 11 variables meteorológicas, debemos pronosticar el caudal de las
48 horas siguientes, en milímetros por hora.

Trabajamos con 508 cuencas: 254 mil ventanas de entrenamiento, 18 mil de
validación y casi 28 mil de test. Las mismas cuencas aparecen en las tres
particiones, pero en periodos distintos. Implementamos DMEL, propuesto por Wang
y colaboradores en *Applied Soft Computing*, en 2026.”

## Slide: ¿Por qué es difícil?

**Tiempo: 0:50–1:35**

“El caudal es muy asimétrico: la mediana es de 0.02 milímetros por hora, pero
hay picos de 15. Su escala cambia tres órdenes de magnitud entre cuencas, no
hay fechas y las crecidas suelen empezar con lluvia que todavía no ha caído.

Como referencia usamos la persistencia, que repite el último caudal observado.
En validación tiene un NSE de 0.998 a una hora, pero cae a 0.58 a 24 horas y a
0.22 a 48 horas. En las primeras horas copiar el último valor es casi
imbatible: el reto está entre las 24 y las 48 horas.”

## Slide: La idea de DMEL

**Tiempo: 1:35–2:10**

“La idea central de DMEL es separar el caudal según su complejidad y darle cada
parte al modelo que mejor la aprende.

Una estrategia de reconocimiento modal, llamada MRS, descompone la serie con
CEEMDAN y mide la complejidad de cada componente con sample entropy. Los
componentes complejos, de alta frecuencia, van a un Informer; los suaves, de
baja frecuencia, a una LSTM. El pronóstico es la suma de ambas ramas.”

## Slide: Paso 1 · CEEMDAN separa el caudal en modos

**Tiempo: 2:10–2:55**

“CEEMDAN es una versión mejorada de la descomposición empírica en modos. Separa
la serie en funciones de modo intrínseco, o IMF, cada una a una escala de
tiempo distinta, más un residuo que representa la tendencia.

Agrega ruido blanco, extrae cada modo y promedia sobre cien realizaciones de
ruido, lo que reduce la mezcla de modos. La suma de los componentes reconstruye
exactamente la señal.

[Señalar la figura.] En esta ventana, el primer IMF captura las variaciones
hora a hora y el residuo la forma de la crecida. El color adelanta la decisión
del MRS.”

## Slide: Paso 2 · Sample entropy mide la complejidad

**Tiempo: 2:55–3:35**

“Para medir la complejidad, el paper usa sample entropy. Se cuenta cuántos
pares de patrones de largo m se parecen y cuántos siguen pareciéndose al
alargarlos a m más uno. Si al alargarlos se pierden muchas coincidencias, la
serie es impredecible y su entropía es alta.

Como referencia, el ruido blanco tiene una entropía de 2.2, un seno de 24 horas
de 0.3 y nuestro caudal horario apenas 0.06: es una serie muy regular.”

## Slide: Paso 3 · MRS: agrupar por complejidad

**Tiempo: 3:35–4:15**

“Con estas entropías, el MRS reagrupa los componentes. Si la entropía de un
componente supera la de la serie original, es de alta frecuencia y va al
Informer; si no, es de baja frecuencia y va a la LSTM.

En nuestro dataset, los cinco primeros IMF quedan como componentes de alta
frecuencia, y el sexto IMF junto con el residuo forman el único componente de
baja frecuencia. Es el mismo patrón que el paper reporta para su estación
Shuangpai.”

## Slide: Paso 4 · Un modelo para cada frecuencia

**Tiempo: 4:15–5:05**

“La hipótesis del paper es que cada tipo de componente necesita un modelo
distinto.

Para la alta frecuencia usa un Informer, un Transformer eficiente para series
largas, con tres ideas: la atención ProbSparse; el distilling, que reduce a la
mitad la longitud entre capas del encoder; y un decoder generativo que produce
las 48 horas en una sola pasada.

Para la baja frecuencia usa una LSTM de dos capas de 256 unidades, cuyas
compuertas conservan la memoria de largo plazo. La predicción final es la suma
de las dos ramas: la ecuación 14 del paper.”

## Slide: ProbSparse: atención solo donde importa

**Tiempo: 5:05–6:00**

“La atención ProbSparse evita calcular la atención completa para todas las
consultas. La actividad de una consulta se mide con la divergencia KL entre su
atención y una atención uniforme, que se reduce a un log-sum-exp menos un
promedio. Pero el log-sum-exp exige todas las claves.

[Señalar la segunda línea de la ecuación.] Los autores del Informer
demostraron que ese término está acotado por el valor máximo: queda entre el
máximo y el máximo más el logaritmo de L. Por eso lo reemplazan por el máximo,
calculado sobre un subconjunto muestreado de claves.

Solo las consultas más activas, c por el logaritmo de L, calculan la atención
completa: con c igual a 5, son 30 de 336 en el encoder. Las demás reciben la
media de los valores.”

---

# 2. Implementación — 6:00 a 12:00

## Slide: ¿Por qué el protocolo del paper filtra el futuro?

**Tiempo: 6:00–7:00**

[Pasar la portada de sección “Implementación” sin detenerse.]

“Al implementar el método encontramos un problema importante en el protocolo
del paper. El paper aplica CEEMDAN a la serie completa, de 2002 a 2005, y recién
después la divide en entrenamiento, validación y test.

La descomposición no es causal: las envolventes que definen cada modo pasan por
los máximos y mínimos a ambos lados de cada instante. Por eso, el valor de un
componente en el tiempo t depende de lo que ocurre después. Al descomponer
antes de dividir, el modelo recibe una pista de los días que debe pronosticar.

En un pronóstico real, y en el test de este laboratorio, esa pista no existe.
Por eso nosotros descomponemos cada ventana usando solo sus 336 horas pasadas.”

## Slide: La fuga, medida

**Tiempo: 7:00–7:45**

“Medimos la fuga en 200 ventanas de validación. Cada ventana se descompone de
dos formas: solo con su historia, como hacemos nosotros, y con la historia más
las 48 horas futuras, como en el paper. Como control, repetimos la
descomposición cambiando solo el ruido.

[Señalar las barras.] Los componentes rápidos casi no cambian, pero los lentos,
que son la entrada de la LSTM, cambian entre cuatro y seis veces más que el
control. [Señalar el ejemplo.] Aquí, el residuo calculado con el futuro sube
antes de la crecida, mientras el caudal sigue en cero: el componente ya sabe lo
que viene.”

## Slide: Nuestro CEEMDAN y MRS

**Tiempo: 7:45–8:35**

“Implementamos CEEMDAN desde cero en numba, con un banco fijo de cien
realizaciones de ruido, una spline cúbica natural propia y diez iteraciones de
sifting por modo. Descomponemos las 300 mil ventanas a 4.6 milisegundos cada
una con 24 CPUs, con un error de reconstrucción de una millonésima.

Como un solo modelo atiende a las 508 cuencas, todas necesitan los mismos
canales. Por eso el MRS se decide una vez, con 8 mil ventanas de
entrenamiento, por votos: un componente es de alta frecuencia si su entropía
supera la de la serie en más de la mitad de las ventanas. Con los valores del
paper, esta regla reproduce exactamente su agrupación.”

## Slide: ¿Qué recibe cada rama?

**Tiempo: 8:35–9:30**

“El Informer recibe 18 canales: los cinco componentes de alta frecuencia, las
11 variables meteorológicas y dos canales estáticos de la cuenca. La LSTM
recibe 14: el componente de baja frecuencia, las covariables y los estáticos.
En la variante, la LSTM recibe además el caudal: 15 canales.

[Señalar la primera ecuación.] Como el caudal varía tres órdenes de magnitud
entre cuencas, dividimos el caudal, los componentes y el objetivo por s sub b,
la desviación estándar de la cuenca más 0.01. La red predice en esas unidades y
multiplicamos por s sub b para volver a milímetros por hora. El error cuadrático
en ese espacio es la pérdida NSE estrella de Kratzert. Los estáticos devuelven
la magnitud que el escalado borra.”

## Slide: Lo que cambiamos del paper

**Tiempo: 9:30–10:05**

“En resumen, descomponemos por ventana para evitar la fuga, agrupamos con votos
globales y entrenamos las ramas sobre la suma, porque los componentes futuros
no existen en inferencia. Además, pasamos de datos diarios a horarios, omitimos
el embedding de fechas y usamos la pérdida NSE estrella con batch de 256.

Conservamos la propuesta central: el MRS con CEEMDAN y sample entropy, el
Informer para la alta frecuencia, la LSTM para la baja y la suma de ambas
ramas.”

## Slide: Cómo entrenamos

**Tiempo: 10:05–10:45**

“Entrenamos con Adam, un learning rate de 3 por 10 a la menos 4 con warmup y
ReduceLROnPlateau. Con batch de 256 hay 993 pasos por época: una época nuestra
equivale a unas 30 del paper. Como el paper, entrenamos 100 épocas y nos
quedamos con la época de menor pérdida de validación. Cada modelo tomó 3.3
horas en una RTX A6000 de KHIPU.

[Señalar las curvas.] La validación deja de mejorar alrededor de la época 10,
mientras el entrenamiento sigue bajando: es sobreajuste.”

## Slide: ¿Cortaba antes de tiempo el early stopping?

**Tiempo: 10:45–11:25**

“Primero entrenamos con early stopping y después repetimos todo con 100 épocas,
para descartar un corte prematuro; la variante solo existe en esta segunda
ronda. Se eligen las mismas épocas, entre la 5 y la 11: el early stopping era
correcto.

Pero hay algo importante. DMEL eligió la misma época, con la misma pérdida de
validación, en las dos rondas, y aun así su NSE de test pasó de 0.629 a 0.614,
por el no determinismo de la GPU. Esa variación es del tamaño de las diferencias
entre modelos.”

## Slide: ¿Qué comparamos?

**Tiempo: 11:25–12:00**

“Comparamos cinco sistemas: la persistencia, como modelo base ingenuo; DMEL, la
propuesta del paper; una ablación que invierte la asignación, con la LSTM en
alta frecuencia y el Informer en baja; otra sin MRS, con la misma arquitectura
pero el caudal crudo en ambas ramas; y una variante propia, que le agrega a la
LSTM de DMEL el caudal observado. Las cuatro redes tienen unos 3.7 millones de
parámetros.”

---

# 3. Resultados — 12:00 a 17:30

## Slide: ¿Cómo medimos?

**Tiempo: 12:00–12:30**

[Pasar la portada de sección “Resultados” sin detenerse.]

“Usamos las métricas del paper. El NSE vale uno si el pronóstico es perfecto y
cero si equivale a predecir el promedio. El KGE combina correlación,
variabilidad y sesgo, y el RMSE mide el error en milímetros por hora. NSE y KGE
se calculan por cuenca y reportamos la mediana. También medimos el error del
pico y usamos Wilcoxon para la significancia.”

## Slide: Resultados en test

**Tiempo: 12:30–13:05**

“En test, DMEL obtiene un NSE mediano de 0.614, frente a 0.492 de la
persistencia, y el Wilcoxon confirma que es mejor en 348 de las 508 cuencas,
con un p-valor del orden de 10 a la menos 19. El RMSE baja de 0.119 a 0.101
milímetros por hora y el error del pico, de 49 a 41 por ciento.

[Señalar la última fila.] Pero las cuatro redes quedan muy cerca entre sí, y
las otras tres superan levemente a DMEL.”

## Slide: ¿Cómo cae el desempeño con el horizonte?

**Tiempo: 13:05–13:35**

“Por horizonte, la persistencia gana hasta unas 6 horas. Desde las 7 horas las
redes la superan y la brecha crece: a 48 horas, el NSE de las redes ronda 0.40,
frente a 0.15 de la persistencia. En las primeras horas dominan los modelos que
ven el caudal crudo, porque las últimas horas de cada componente son las menos
estables.”

## Slide: Las ablaciones

**Tiempo: 13:35–14:15**

“En las ablaciones, invertir la asignación de frecuencias da un NSE levemente
mejor que DMEL, y quitar el MRS también. En la primera ronda pasó lo mismo:
DMEL nunca supera a sus ablaciones.

El Wilcoxon es significativo porque compara 508 pares de cuencas, pero compara
dos modelos entrenados, no dos métodos. Con una variación entre corridas de
0.01 a 0.015, lo robusto es la dirección: ni el MRS ni la asignación de
frecuencias aportan una mejora medible.”

## Slide: ¿Por qué el MRS no ayuda aquí?

**Tiempo: 14:15–14:50**

“Vemos tres razones. Primero, CEEMDAN es una transformación determinista de la
misma historia: no agrega información, y por ventana degrada justo las horas
más recientes. Segundo, el protocolo del paper se beneficia de la fuga que
medimos. Y tercero, el paper es univariado, con siete días de entrada, mientras
que nuestras redes también ven la lluvia.”

## Slide: Variante: devolverle el caudal a DMEL

**Tiempo: 14:50–15:25**

“Como DMEL perdía en las primeras horas, probamos agregarle a la LSTM el caudal
observado. El NSE a una hora sube de 0.943 a 0.984, y a seis horas de 0.907 a
0.948, superando a la persistencia. También logra el mejor KGE, 0.693, y el
menor error de pico. Pero su NSE mediano iguala al modelo sin MRS: la ganancia
viene del caudal crudo, no de la descomposición.”

## Slide: ¿Qué modelo es mejor?

**Tiempo: 15:25–15:50**

“En resumen: la persistencia solo gana en las primeras seis horas, DMEL es la
red más débil, el modelo sin MRS tiene el mejor NSE y la variante es la más
equilibrada. Las diferencias entre redes caben en la variación entre corridas;
la brecha con la persistencia, en cambio, es grande y robusta.”

## Slide: Las crecidas

**Tiempo: 15:50–16:15**

“En las mayores crecidas el caudal parte casi de cero y ningún modelo las
anticipa: la lluvia que las provoca todavía no ha caído y no tenemos
meteorología futura. Cuando la crecida ya empezó, como en la cuenca 363, la
variante es la que más se acerca al pico.”

## Slide: ¿Qué aporta cada rama?

**Tiempo: 16:15–16:35**

“Las ramas de DMEL no se reparten las frecuencias: se compensan con
desplazamientos casi constantes, y la crecida la explica la LSTM, que también
recibe la lluvia. La interpretación del paper no se sostiene aquí.”

## Slide: ¿Dónde falla?

**Tiempo: 16:35–17:00**

“En caudales bajos, la persistencia tiene un error relativo del 10 por ciento y
las redes de 47 a 68: las recesiones lentas se predicen mejor copiando el
último valor. Y en las crecidas, todos subestiman el pico entre un 30 y un 40
por ciento.”

## Slide: Limitaciones

**Tiempo: 17:00–17:10**

“Nuestras limitaciones son el efecto de borde de la descomposición, una sola
agrupación para todas las cuencas, una sola semilla y la falta de lluvia
futura.”

## Slide: Conclusiones

**Tiempo: 17:10–17:30**

“En conclusión: implementamos DMEL desde cero y mostramos que descomponer la
serie completa filtra información del futuro. En un pronóstico honesto, DMEL
supera a la persistencia en los horizontes largos, pero sus componentes
centrales no mejoran a la misma red con el caudal crudo. Gracias.”

---

# Apéndices — solo para preguntas

No narrar durante el video principal:

- `Apéndice: configuración completa`
- `Apéndice: todas las métricas de test`
- `Apéndice: entrenamiento`
- `Apéndice: NSE de las cuencas por horizonte`
- `Apéndice: desempeño vs magnitud del caudal`
- `Apéndice: MRS en números`
- `Apéndice: distribución del NSE por cuenca`
- `Apéndice: observado vs pronosticado a 24 h`
- `Apéndice: código de la implementación`

# Referencia principal

Wang, W.-c., Gu, M., Li, Z., Wang, Y.-y. y Zhao, Y.-w. (2026). *DMEL: A novel
dual-modal ensemble learning architecture for multi-step runoff prediction*.
*Applied Soft Computing*, 192, 114792.
<https://doi.org/10.1016/j.asoc.2026.114792>
