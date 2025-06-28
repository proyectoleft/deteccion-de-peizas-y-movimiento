# Sistema de Detección y Manipulación de Piezas de ajedrez

## Objetivos    
- Detectar las piezas y registrar la posición de todas las piezas en tiempo real.  
- Virtualizar la partida de ajedrez el tablero físico y una representación digital sincronizada.  
- Implementar validaciones para evitar movimientos erróneos y mostrar mensajes de advertencia.
## Metodologia incremental
El proyecto sigue un enfoque incremental, donde el sistema se desarrolla en incrementos funcionales definidos. 
Cada incremento agrega una nueva funcionalidad al sistema, construyendo sobre lo que se ha implementado previamente, hasta completar el producto final
justificacion:
- Construcción Progresiva de Funcionalidades:El enfoque incremental permite dividir el proyecto en partes manejables que se desarrollan y entregan de manera secuencial
- Validación y Estabilidad en Cada Incremento: Cada incremento produce un sistema funcional que puede ser probado y validado antes de pasar al siguiente.
- Planificación Estructurada y Predecible:La metodología incremental permite una planificación clara y estructurada, donde cada incremento tiene un objetivo definido
- Integración y Mejora Continua:Cada incremento se integra con los anteriores, permitiendo construir un sistema más completo en cada etapa.

## Planificación de 8 Semanas - Modelo Incremental

### Semana 1: Configuración de Captura de Video

- **Descripción:** Implementar la captura de video en vivo desde una IP Webcam con resolución 720x480.
- **Especificación:** Especificar la conexión a IP Webcam y la visualización del video con FPS.
- **Desarrollo:**
  - Versión Inicial: Configurar `camara.py` para conectar a IP Webcam y capturar video.
  - Versiones Intermedias: Ajustar resolución, manejar fallos de conexión y mostrar FPS en una ventana.
  - Versión Final: Entregar un módulo estable de captura de video con visualización funcional.
- **Validación:** Validar que la cámara se inicialice (`cap.isOpened()`), lea frames y muestre el video correctamente al ejecutar `camara.py`.

### Semana 2: Detección de Piezas

- **Descripción:** Integrar detección en tiempo real de piezas de ajedrez usando el modelo YOLO.
- **Especificación:** Definir la integración de `deteccion.py` con `camara.py` y la visualización de detecciones.
- **Desarrollo:**
  - Versión Inicial: Implementar detección de piezas con YOLO usando un frame de prueba.
  - Versiones Intermedias: Optimizar con delay para mejorar FPS y dibujar recuadros en el video.
  - Versión Final: Entregar detección funcional integrada, mostrando piezas detectadas en tiempo real.
- **Validación:** Validar que el modelo `best.pt` detecte piezas en el video con confianza adecuada.

### Semana 3: Detección de Esquinas

- **Descripción:** Permitir al usuario marcar manualmente las esquinas del tablero y guardar sus coordenadas.
- **Especificación:** Especificar `corners.py` para permitir al usuario marcar esquinas y guardar coordenadas.
- **Desarrollo:**
  - Versión Inicial: Implementar una interfaz básica para marcar esquinas en un frame.
  - Versiones Intermedias: Mejorar la interfaz para mostrar y guardar coordenadas de esquinas.
  - Versión Final: Entregar un módulo que detecte y almacene esquinas de manera confiable.
- **Validación:** Validar que las esquinas marcadas sean precisas y se guarden correctamente en un frame real.

### Semana 4: Transformación de Coordenadas

- **Descripción:** Mapear las coordenadas de píxeles detectados a casillas reales del tablero (a1–h8).
- **Especificación:** Definir `transform.py` para transformar coordenadas usando las esquinas detectadas.
- **Desarrollo:**
  - Versión Inicial: Desarrollar una transformación básica con coordenadas fijas.
  - Versiones Intermedias: Ajustar para diferentes tamaños de tablero y mejorar precisión.
  - Versión Final: Entregar un módulo que mapee coordenadas de manera precisa.
- **Validación:** Validar que las coordenadas de detecciones se mapeen correctamente a casillas del tablero.

### Semana 5: Tablero Virtual

- **Descripción:** Crear un tablero digital que refleje la posición de las piezas detectadas en el video.
- **Especificación:** Especificar `virtual_board.py` para mostrar un tablero con piezas detectadas.
- **Desarrollo:**
  - Versión Inicial: Implementar un tablero básico con piezas iniciales usando `chess`.
  - Versiones Intermedias: Integrar detecciones y actualizar el tablero dinámicamente.
  - Versión Final: Entregar un tablero virtual funcional y sincronizado con el video.
- **Validación:** Validar que el tablero virtual refleje las piezas detectadas en el video físico.

### Semana 6: Mejora del Tablero Virtual con python-chess 
- **Descripción:** Reemplazar el tablero virtual de OpenCV por uno basado en python-chess, optimizando la visualización y sincronización.
- **Especificación:** Especificar virtual_board.py para usar python-chess, integrando detecciones y mejorando estabilidad.
- **Desarrollo:**
  - Versión Inicial: Implementar un tablero con python-chess, mostrando piezas detectadas.
  - Versiones Intermedias: Optimizar sincronización (usando caché y estabilidad), mejorar FPS y ajustar parámetros como STABILITY_THRESHOLD.
  - Versión Final: Entregar un tablero virtual robusto con python-chess, superando las limitaciones del tablero OpenCV.
- **Validación:** Validar que el tablero con python-chess sea más preciso y eficiente (FPS, estabilidad) que el de OpenCV, reflejando correctamente las detecciones.

### Semana 7: Documentación y Preparación de Entrega

- **Descripción:** Redactar y organizar la documentación técnica del proyecto para su presentación final.
- **Especificación:** Especificar la documentación del código, metodología y resultados.
- **Desarrollo:**
  - Versión Inicial: Documentar el código y los incrementos realizados.
  - Versiones Intermedias: Revisar y ajustar la documentación.
  - Versión Final: Entregar la documentación completa.
- **Validación:** Validar que la documentación sea clara y completa.

### Semana 8: Entrega Final y Revisión

- **Descripción:** Realizar la entrega formal del proyecto, incluyendo revisión funcional y presentación.
- **Especificación:** Definir la presentación o entrega del proyecto completo.
- **Desarrollo:**
  - Versión Inicial: Preparar una presentación o entrega preliminar.
  - Versiones Intermedias: Ajustar basándose en retroalimentación inicial.
  - Versión Final: Entregar el proyecto completo con documentación final.
- **Validación:** Validar con una presentación o entrega formal ante usuarios o revisores.



