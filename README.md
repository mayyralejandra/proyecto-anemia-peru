# 🩸 Clasificación de Anemia en Niños — Proyecto Integrador

## Descripción

Sistema de Machine Learning que predice el diagnóstico de anemia (Normal, Anemia Leve, Anemia Moderada, Anemia Severa) en niños menores de 5 años, usando datos clínicos y socioeconómicos del Sistema de Información del Estado Nutricional (SIEN) del departamento de Ayacucho, Perú.

**Tipo de problema:** Clasificación multiclase  
**Variable objetivo:** `Dx_anemia`  
**Modelo final:** Random Forest (ajustado con GridSearchCV + SMOTE)

---

## Estructura del proyecto

```
proyecto/
│
├── app/
│   └── app.py              # Aplicación web Streamlit
│
├── datos/
│   └── anemia.csv          # Dataset original (35,967 registros)
│
├── models/
│   └── modelo_final.pkl    # Modelo entrenado (generado por el notebook)
│
├── notebooks/
│   └── modelo_final.ipynb  # Notebook completo del proyecto
│
├── informe/                # Carpeta para informes adicionales
│
├── README.md
└── requirements.txt
```

---

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución del Notebook

```bash
cd notebooks
jupyter notebook modelo_final.ipynb
```

Ejecutar todas las celdas en orden. Al finalizar, el modelo se guardará automáticamente en `models/modelo_final.pkl`.

> **Importante:** Ejecutar el notebook antes de lanzar la app, para generar el archivo `.pkl`.

---

## Ejecución de la aplicación Streamlit

```bash
cd app
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

---

## Variables utilizadas

| Variable       | Descripción                        | Tipo       |
|----------------|------------------------------------|------------|
| Sexo           | Sexo del paciente (M/F)            | Binaria    |
| EdadMeses      | Edad en meses                      | Numérica   |
| AlturaREN      | Altitud del lugar de residencia    | Numérica   |
| Hbc            | Hemoglobina corregida por altitud  | Numérica   |
| Cred           | Control CRED al día                | Binaria    |
| Suplementacion | Recibe suplemento de hierro        | Binaria    |
| Juntos         | Beneficiario del programa Juntos   | Binaria    |
| SIS            | Tiene Seguro Integral de Salud     | Binaria    |
| Qaliwarma      | Beneficiario de Qali Warma         | Binaria    |

---

## Métricas del modelo final

| Métrica    | Valor (aprox.) |
|------------|----------------|
| Accuracy   | ~97%           |
| F1 Macro   | ~85%           |
| F1 Normal  | ~99%           |
| F1 A. Leve | ~85%           |
| F1 A. Mod. | ~70%           |

> Los valores exactos se obtienen al ejecutar el notebook.

---

## Notas

- El dataset presenta desbalance de clases significativo (Anemia Severa: solo 11 casos).
- Se aplicó SMOTE para balancear el conjunto de entrenamiento.
- La clase Anemia Severa tiene limitaciones por escasez de datos.
- El modelo es orientativo y no reemplaza el diagnóstico clínico profesional.
