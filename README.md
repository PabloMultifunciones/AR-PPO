# AR-PPO
Politica de Optimizacion (PPO)
### Introduccion ###

En 2018, OpenAI hizo un gran avance en el aprendizaje de refuerzo profundo, esto fue posible solo gracias a una arquitectura de hardware sólida y al uso del algoritmo de última generación: optimización de política próxima.

La idea principal de la optimización de políticas próxima es evitar tener una actualización de políticas demasiado grande. Para hacer eso, usamos una relación que nos dice la diferencia entre nuestra póliza nueva y la anterior y recortamos esta relación de 0.8 a 1.2. Al hacerlo, se asegurará de que la actualización de la política no sea demasiado grande.

### Problema con el gradiente de política ###

La funcion de perdida de la politica del gradiente (Policy Loss) es la siguiente:  

<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198849592-02065d58-e490-4e8a-b6d9-58f4f3aaa4d5.svg">
</p>

L(Q) - Pérdida de póliza  
E - Esperado  
logπ... - probabilidad logarítmica de realizar esa acción en ese estado  
A - Ventaja  

La idea de esta función es hacer un paso de ascenso de gradiente (que es equivalente a tomar un descenso de gradiente inverso). De esta manera, nuestro agente se ve obligado a tomar acciones que lo lleven a obtener mayores recompensas y evitar acciones dañinas.

Sin embargo, el problema viene del tamaño del paso:  
* Demasiado pequeño, el proceso de formación es demasiado lento  
* Demasiado alto, hay demasiada variabilidad en el entrenamiento  

Ahí es donde la PPO es útil; la idea es que PPO mejore la estabilidad de la capacitación de Actor al limitar la actualización de la política en cada paso de capacitación.

Para hacer eso, PPO introdujo una nueva función de objetivo llamada "Función de objetivo sustituta recortada" que restringirá el cambio de política en un rango pequeño usando un clip.

### Función de objetivo sustituto recortada ###

Primero, como se explica en el documento de PPO, en lugar de usar log pi para rastrear el impacto de las acciones, PPO usa la relación entre la probabilidad de acción bajo la política actual dividida por la probabilidad de acción bajo la política anterior:

<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198849914-7c82b0ca-2cf9-42d4-b90a-077180774e46.svg">
</p>

Como podemos ver, rt(Q) es la razón de probabilidad entre la póliza nueva y la vieja:  
  
* Si rt(Q) >1, la acción es más probable en la política actual que en la anterior.  
* Si rt(Q) está entre 0 y 1, la acción es menos probable para la política actual que para la antigua.  
  
Como consecuencia, podemos escribir nuestra nueva función objetivo:  

<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198853047-eff82ca7-729a-4961-98f6-b459627d1946.svg">
</p>

Sin embargo, si la acción fuera más probable en nuestra política actual que en la anterior, esto conduciría a un paso de gradiente de política gigante y una actualización de política excesiva.

En consecuencia, necesitamos restringir esta función objetivo penalizando los cambios que conducen a una razón (en el artículo, se dice que la razón solo puede variar de 0,8 a 1,2). Para hacer eso, tenemos que usar la relación de probabilidad de recorte PPO directamente en la función objetivo con su función objetivo sustituta recortada. Al hacerlo, nos aseguraremos de no tener una actualización de política demasiado grande porque la nueva política no puede ser muy diferente de la anterior.

<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198853067-84ee1ed2-d3f0-466f-a0b5-844fc8139878.svg">
</p>

Con la función Clipped Surrogate Objective, tenemos dos razones de probabilidad, una no recortada y otra recortada en un rango (entre [1−∈,1+∈], epsilon es un hiperparámetro que nos ayuda a definir este rango de recorte (en el papel ∈ = 0,2).

Si tomamos el mínimo de los objetivos recortados y no recortados, el objetivo final sería más bajo (límite pesimista) que la meta no recortada. En consecuencia, tenemos dos casos a considerar:

<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198853177-b3ff319a-0b5c-451d-86e7-c11eb1e10ecf.png">
</p>

* Cuando la ventaja es > 0:
  
Si ventaja > 0, la acción es mejor que el promedio de todas las acciones en ese estado. Por lo tanto, debemos fomentar nuestra nueva política para aumentar la probabilidad de realizar esa acción en ese estado.
  
Esto significa aumentar rt porque aumentamos la probabilidad en la nueva política y el denominador de la política anterior permanece constante:  

<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198853235-9c8365e0-dc7e-42ba-8a51-38bc8e6c1ce2.svg">
</p>
  
Sin embargo, debido al clip, rt(Q) solo crecerá hasta 1+∈, lo que significa que esta acción no puede ser cien veces más probable en comparación con la política anterior (debido al clip). Esto se hace porque no queremos actualizar demasiado nuestra política. Tomar acción en un estado específico es solo un intento. No significa que siempre conducirá a una recompensa súper positiva, por lo que no queremos ser demasiado codiciosos porque también puede conducir a una mala política.
  
En resumen, en el caso de ventaja positiva, queremos aumentar la probabilidad de realizar esa acción en ese paso, pero no demasiado.

* Cuando la ventaja es < 0:
  
Si la ventaja < 0, la acción debe desanimarse debido al efecto negativo del resultado. En consecuencia, rt disminuirá (porque la acción es menos probable para la política de agente actual que para la anterior), pero rt solo disminuirá hasta 1−∈ debido al recorte.

Además, no queremos hacer un gran cambio en la política siendo demasiado codiciosos al reducir en última instancia la probabilidad de tomar esa acción porque conduce a una ventaja opuesta.

En resumen, gracias a este objetivo sustituto recortado, el rango en el que la nueva política puede variar con respecto a la anterior se restringe porque se elimina el incentivo para que la razón de probabilidad se mueva fuera del intervalo. Si la relación es > 1+e o < 1-e, el gradiente será igual a 0 (sin pendiente) porque el clip tiene el efecto de un gradiente. Por lo tanto, estas dos regiones de recorte no nos permiten volvernos demasiado codiciosos e intentar actualizar demasiado a la vez y actualizar fuera de la región donde este ejemplo tiene una buena aproximación.
  
La última pérdida de objetivo sustituta recortada:
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/95035101/198853429-a4fe38d4-7a57-4dfa-af37-36b2f6ae0875.svg">
</p>

* c1 y c2 - coeficientes  
* S - denota una bonificación de entropía  
* Lv y Ft - pérdida de error al cuadrado  
