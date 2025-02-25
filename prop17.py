import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML
import re

# Configurar página para usar todo el ancho y añadir CSS personalizado
st.set_page_config(layout="wide")

# CSS personalizado para centrar y dar estilo
st.markdown("""
    <style>
        .main {
            padding: 0 !important;
        }
        .stPlotlyChart {
            margin: 0 auto;
            display: block !important;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Funciones geométricas básicas
def dibujar_punto(fig, x, y, nombre=None):
    """Dibuja un punto con etiqueta opcional"""
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        text=[nombre] if nombre else None,
        textposition='top center',
        marker=dict(size=15),  # Tamaño del punto
        textfont=dict(size=40)  # Tamaño de la etiqueta
    ))

def dibujar_segmento(fig, x1, y1, x2, y2):
    """Dibuja un segmento entre dos puntos"""
    fig.add_trace(go.Scatter(
        x=[x1, x2], y=[y1, y2],
        mode='lines',
        line=dict(width=2)
    ))



def dibujar_circulo(fig, centro, radio_punto):
    #Dibuja un círculo dado su centro y un punto en el borde
    x_centro, y_centro = centro
    x_radio, y_radio = radio_punto
    
    # Calcular radio (distancia entre los puntos)
    radio = np.sqrt((x_radio - x_centro)**2 + (y_radio - y_centro)**2)
    
    # Generar puntos del círculo
    theta = np.linspace(0, 2*np.pi, 100)
    x = x_centro + radio * np.cos(theta)
    y = y_centro + radio * np.sin(theta)
    
    # Añadir el círculo a la figura
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(width=2, color='blue')
    ))
    
    # Ajustar los ejes para que el círculo sea visible (¡clave aquí!)
    fig.update_layout(
        xaxis=dict(range=[x_centro - radio - 1, x_centro + radio + 1]),
        yaxis=dict(range=[y_centro - radio - 1, y_centro + radio + 1]),
        yaxis_scaleanchor="x"  # Relación de aspecto 1:1
    )


def dibujar_angulo(fig, A, B, C, radio=0.5, nombre=None):
    """
    Dibuja el ángulo ∠ABC en un gráfico de Plotly.

    Parámetros:
    - fig: Figura de Plotly.
    - A, B, C: Tuplas (x, y) con las coordenadas de los puntos.
    - radio: Radio del arco del ángulo (por defecto 0.5).
    - nombre: Etiqueta opcional para el ángulo.
    """
    # Convertir puntos a numpy arrays
    A, B, C = np.array(A), np.array(B), np.array(C)
    
    # Vectores BA y BC
    BA = A - B
    BC = C - B
    
    # Ángulos en radianes
    angulo_inicial = np.arctan2(BA[1], BA[0])
    angulo_final = np.arctan2(BC[1], BC[0])
    
    # Asegurar que el arco es en sentido correcto
    if angulo_final < angulo_inicial:
        angulo_final += 2 * np.pi
    
    # Generar puntos para el arco
    theta = np.linspace(angulo_inicial, angulo_final, 50)
    x = B[0] + radio * np.cos(theta)
    y = B[1] + radio * np.sin(theta)
    
    # Dibujar el arco
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict( width=2)))
    
    # Etiqueta opcional
    if nombre:
        angulo_medio = (angulo_inicial + angulo_final) / 2
        x_text = B[0] + (radio + 0.1) * np.cos(angulo_medio)
        y_text = B[1] + (radio + 0.1) * np.sin(angulo_medio)
        fig.add_trace(go.Scatter(
            x=[x_text], y=[y_text],
            text=[nombre], mode='text',
            textposition='middle center'
        ))
 

# Diccionario de funciones disponibles
FUNCIONES_DISPONIBLES = {
    'punto': dibujar_punto,
    'segmento': dibujar_segmento,
    'circulo': dibujar_circulo,
    'angulo': dibujar_angulo
}

# Configuración básica del gráfico
def crear_figura_base():
    """Crea una figura base con la configuración correcta"""
    fig = go.Figure()
    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
        width=1000,  # Figura más grande
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    # Asegurar que la proporción de aspecto sea 1:1
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


    
    # Añadir etiqueta si se proporciona
    if nombre:
        # Calcular posición para la etiqueta (en el punto medio del arco)
        ang_medio = (ang_inicial + ang_final) / 2
        label_x = x_vertice + (radio * 1.2) * np.cos(ang_medio)
        label_y = y_vertice + (radio * 1.2) * np.sin(ang_medio)
        
        fig.add_trace(go.Scatter(
            x=[label_x], y=[label_y],
            mode='text',
            text=[nombre],
            textposition='middle center'
        ))


def ejecutar_paso(fig, funcion_nombre, parametros):
    """Ejecuta una función geométrica con sus parámetros"""
    funcion = FUNCIONES_DISPONIBLES.get(funcion_nombre)
    if funcion:
        params = eval(parametros) if isinstance(parametros, str) else parametros
        funcion(fig, **params)

def cargar_pasos_excel():
    """Carga los pasos desde Excel o crea un DataFrame de ejemplo"""
    try:
        return pd.read_excel(""https://raw.githubusercontent.com/jpmcmoreno/Proposicion-17/main/pruebaV2.xlsx"")
    except:
        return pd.DataFrame({
            'paso': [1, 2, 3, 4],
            'descripcion': [
                'Dibujar puntos A y B',
                'Trazar círculo con centro en A',
                'Trazar círculo con centro en B',
                'Conectar punto C'
            ],
            'justificacion': [
                'Postulado 1: Puntos iniciales',
                'Postulado 3: Círculo con centro A',
                'Postulado 3: Círculo con centro B',
                'Intersección de círculos'
            ],
            'funcion': [
                'punto',
                'circulo',
                'circulo',
                'punto'
            ],
            'parametros': [
                "{'x': -1, 'y': 0, 'nombre': 'A'}",
                "{'x_centro': -1, 'y_centro': 0, 'radio': 2}",
                "{'x_centro': 1, 'y_centro': 0, 'radio': 2}",
                "{'x': 0, 'y': 1.732, 'nombre': 'C'}"
            ]
        })


proposiciones_euclides = pd.read_excel("https://raw.githubusercontent.com/jpmcmoreno/Proposicion-17/main/proposiciones_euclides.xlsx")
def invocar_descripcion(nombre, df):
    resultado = df.loc[df['Proposición'] == nombre, 'Descripción']
    return resultado.iloc[0] if not resultado.empty else "Proposición no encontrada"

def main():
    st.title("Elementos de Euclides - Proposición I.XVII")

    st.markdown(f"## {invocar_descripcion("Proposición I.XVII", proposiciones_euclides)}")
    
    # Cargar datos desde Excel
    df_pasos = cargar_pasos_excel()
    
    if 'paso_actual' not in st.session_state:
        st.session_state.paso_actual = 1

    # Dividir en dos columnas con proporción 1:3
    col_info, col_grafico = st.columns([1, 3])
    
    with col_info:
        st.markdown("### Control de Pasos")
        # Controles en una sola fila
        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀ Anterior", use_container_width=True) and st.session_state.paso_actual > 1:
                st.session_state.paso_actual -= 1
        with col2:
            if st.button("Siguiente ▶", use_container_width=True) and st.session_state.paso_actual < len(df_pasos):
                st.session_state.paso_actual += 1
        
        st.markdown("---")
        # Información del paso actual
        paso_actual = df_pasos.iloc[st.session_state.paso_actual - 1]

        
        #st.table(df_pasos[['paso', 'descripcion', 'justificacion']].iloc[:st.session_state.paso_actual])
        # Crear subset de datos
        # st.dataframe(df_subset.style.hide(axis="index"))
        df_subset = df_pasos[['paso', 'descripcion', 'justificacion']].iloc[:st.session_state.paso_actual] \
            .rename(columns={'paso': 'Paso', 'descripcion': 'Descripción', 'justificacion': 'Justificación'}) \
            .iloc[::-1]  # Invierte el orden de las filas

        st.markdown("""
            <style>
            table {
                font-size: 20px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Crear la tabla en texto Markdown manualmente
        tabla_md = "|Paso | Afirmación | Razón |\n"
        tabla_md += "|------|------------|---------------|\n"

        for _, row in df_subset.iterrows():
            tabla_md += f"| {row['Paso']} | {row['Descripción']} | {row['Justificación']} |\n"

        # Mostrar la tabla en Streamlit
        st.markdown(tabla_md)


    with col_grafico:

        justificacion_actual = paso_actual["justificacion"]

        # Expresión regular para detectar "Proposición X.Y"
        match = re.search(r"Proposición\s+([IVXLCDM]+)\.([IVXLCDM]+)", justificacion_actual)

        if match:
            nombre_proposicion = match.group()  # Extrae el nombre completo "Proposición I.12"
            descripcion = invocar_descripcion(nombre_proposicion, proposiciones_euclides)
            st.markdown(f"## 📝 Nota: {nombre_proposicion}\n ### {descripcion}")

        # Crear y mostrar figura
        fig = crear_figura_base()
        for i in range(st.session_state.paso_actual):
            paso = df_pasos.iloc[i]
            ejecutar_paso(fig, paso['funcion'], paso['parametros'])
        st.plotly_chart(fig, use_container_width=True)
        
       

if __name__ == "__main__":
    main()



