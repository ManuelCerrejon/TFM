# Descripción del Repositorio

Este repositorio contiene el código fuente, notebooks, modelos y recursos necesarios para replicar el experimento de fine-tuning del modelo Whisper utilizando técnicas de **Low-Rank Adaptation (LoRA)**. El trabajo se enmarca dentro del **Trabajo de Fin de Máster (TFM)** y tiene como objetivo explorar métodos de adaptación eficiente de modelos de transcripción automática en contextos específicos del idioma español.

El enfoque principal se centra en reducir los recursos computacionales necesarios para el entrenamiento de modelos grandes mediante el uso de LoRA, una técnica que permite actualizar únicamente un subconjunto de parámetros del modelo base. El proyecto hace uso del modelo Whisper de OpenAI, integrando herramientas de la librería Hugging Face para su adaptación.

---

# Contenido del Repositorio

El repositorio está organizado en distintas carpetas y archivos, cada uno con un propósito específico dentro del flujo de trabajo:

### Cuaderno/
Contiene los notebooks de desarrollo utilizados durante el proyecto. En particular, el notebook `fine_tuneWhisperAdaptado‑LORA‑v3‑DEFINITIVO.ipynb` documenta el proceso completo de fine-tuning: desde la preparación de los datos, la configuración del entorno, hasta la evaluación final del modelo ajustado.

### Datos/
Carpeta destinada a almacenar los conjuntos de datos utilizados para entrenamiento y evaluación.

### Memoria/
Documento con todo el contenido desarrollado en la experimentación.

### Modelo/
Modelo entrenado y checkpoints generado a lo largo del proceso de fine-tuning.

### Simulador/
Simulador creado para probar la transcripción automática




