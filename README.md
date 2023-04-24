# Buscador de palabras de texto por similitud usando GloVe - Pytorch
Trabajar con palabras es siempre un desafío. Incluso para un corpus pequeño, su red neuronal (o cualquier tipo de modelo) debe admitir miles de entradas y salidas discretas. Aparte de las palabras numéricas sin procesar, la técnica estándar de representar palabras como matrices de un solo valor (por ejemplo, "uno"=[00010000...]) no captura ninguna información sobre las relaciones entre las palabras.

Las incrustaciones de palabras (word embeddings) resuelven este problema al representar palabras en un espacio vectorial multidimensional, lo que puede llevar la dimensionalidad del problema de cientos de miles a solo cientos. Además, el espacio vectorial es capaz de capturar relaciones semánticas entre palabras en términos de distancia y aritmética vectorial. Existen algunas técnicas para crear vectores de palabras.

El algoritmo Word2vec predice palabras en contexto (por ejemplo, cuál es la palabra más probable en la oración "Ogato_____en el techo"), mientras que los vectores GloVe se basan en recuentos globales en todo el corpus. Usaremos GloVe ahora para este estudio de caso.

# Modelo GloVe
GloVe es una técnica de vector de palabras (incrustaciones). Los vectores de palabras colocan las palabras en un espacio vectorial, donde las palabras similares se agrupan y las palabras diferentes se repelen entre sí. La ventaja de GloVe es que, a diferencia de Word2vec, GloVe no solo se basa en estadísticas locales (información de contexto local de palabras), sino que incorpora estadísticas globales (co-ocurrencia de palabras) para obtener vectores de palabras. Pero hay mucha sinergia entre GloVe y Word2vec.

Y no se sorprenda al saber que la idea de usar estadísticas globales para derivar relaciones semánticas entre palabras se remonta a mucho tiempo atrás. GloVe significa "Global Vectors" o "Vectores Globales". Y, como se mencionó anteriormente, GloVe captura estadísticas globales y estadísticas locales de un corpus para crear matrices de palabras. Pero, ¿necesitamos estadísticas globales y locales?

Resulta que cada tipo de estadística tiene su propia ventaja. Por ejemplo, Word2vec, que captura estadísticas locales, funciona muy bien en tareas de analogía. Sin embargo, un método como LSA (Latent Semantic Analysis), que usa solo estadísticas globales, no funciona bien en tareas de analogía.

## ¿Cómo funciona GloVe?
Dado un corpus con V palabras, la matriz de coocurrencia X será una matriz V x V, donde la i-fila y la j-ésima columna de X, X_ij indica cuántas veces la palabra i co-ocurrió con la palabra j. Un ejemplo de una matriz de co-ocurrencia podría verse así:

![imagem_2023-04-24_163206059](https://user-images.githubusercontent.com/97414922/234097258-96f1a0f2-139a-4802-9bf7-8c0045eac404.png)


¿Cómo obtenemos una métrica que mide la similitud semántica entre palabras? Para esto necesitamos tres palabras a la vez. Permítanme presentar concretamente esta declaración:

![imagem_2023-04-24_163326381](https://user-images.githubusercontent.com/97414922/234097545-79a2bb32-9c3f-41f1-93b3-bb36782a37cd.png)

Donde: ***P_ik / P_jk donde P_ik = X_ik / X_i***

Aquí P_ik denota la probabilidad de ver las palabras i y k juntas, que se calcula dividiendo el número de veces que i y k aparecieron juntos (X_ik) por el número total de veces que las palabras aparecieron en el corpus (X_i).

Puede ver que, dadas dos palabras, es decir, ice (hielo) y steam (vapor), si la tercera palabra k es muy similar a hielo pero irrelevante para vapor (por ejemplo, k = moda), P_ik / P_jk será muy alto (> 1), y muy similar al vapor, pero irrelevante para el hielo (por ejemplo, k = gas), P_ik / P_jk será muy pequeño (< 1), y si está relacionado o no con cualquiera de las palabras, P_ik / P_jk estará cerca de 1.

Entonces, si podemos encontrar una manera de incorporar P_ik / P_jk en la computación de vectores de palabras, lograremos el objetivo de usar estadísticas globales al aprender vectores de palabras.

Y eso es exactamente lo que hace el modelo GloVe. Enlace a continuación -> https://nlp.stanford.edu/pubs/glove.pdf

## Fuente de datos

Para este caso de estudio, utilizaremos el famoso texto de Isaac Asimov:The Last Question.

Origen del archivo -> http://users.ece.cmu.edu/~gamvrosi/thelastq.html

Traducimos el texto y lo usamos para entrenar el modelo GloVe y luego buscamos palabras por similitud. Recomendado para leer el archivo asimov.txt. 

## Fin