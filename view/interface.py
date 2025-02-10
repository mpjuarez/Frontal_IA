import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model.UrlClassifier import UrlClassifier

def bar_chart(df):
    pred_count = df['type'].value_counts().reset_index()
    pred_count.columns = ['type', 'cantidad'] 
    pred_count['porcentaje'] = (pred_count['cantidad'] / pred_count['cantidad'].sum()) * 100
    pred_count['texto'] = pred_count.apply(lambda row: f"{row['cantidad']} ({row['porcentaje']:.2f}%)", axis=1)

    fig = px.bar(pred_count, 
                    x='type', 
                    y='cantidad', 
                    title='Cantidad de registros por prediccion',
                    labels={'cantidad': 'Cantidad de Registros', 'type': 'predicción'}, 
                    color='type',
                    text='texto'
                )

    st.plotly_chart(fig)
    if "EDIFICIO" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='EDIFICIO', hue='type', data= df)

        # Personalización del gráfico
        plt.title('Predicciones por Edificio')
        plt.xlabel('Edificio')
        plt.ylabel('Cantidad de Predicciones')
        plt.xticks(rotation=45, ha='right')  # Rotar etiquetas del eje X para mejor lectura
        plt.legend(title="Categoría")  # Agregar leyenda con el título adecuado
        plt.tight_layout()  # Ajustar diseño para evitar superposición

        st.pyplot(plt)
    else:
        st.error("No se puede graficar predecciones por EDIFICIO (no existe la columna en el csv)")

    if "jornada" in df.columns:
        count_data = df.groupby(['jornada', 'type']).size().reset_index(name='cantidad')

        total_por_jornada = count_data.groupby('jornada')['cantidad'].transform('sum')
        count_data['porcentaje'] = (count_data['cantidad'] / total_por_jornada) * 100

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='jornada', y='porcentaje', hue='type', data=count_data)

        # Agregar etiquetas de porcentaje a cada barra
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # Evitar etiquetas en barras de 0%
                ax.annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

        # Personalización del gráfico
        plt.title('Predicciones por Jornada en Porcentaje')
        plt.xlabel('Jornada')
        plt.ylabel('Porcentaje de Predicciones')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Categoría")
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.error("No se puede graficar predecciones por jornada (no existe la columna en el csv)")

def view_interface():
    urlClassifier = UrlClassifier()
    st.title("Clasificador de URL")
    url = st.text_input("Ingresa la URL:")

    if url:
        model_name = st.selectbox("Selecciona el modelo:", urlClassifier.get_models(), key= "select_box_input") 
        if st.button("Predecir", key= "btn_input"):
            prediction, probability = urlClassifier.predic(url, model_name)
            st.write(f"Predicción: {prediction}")
            st.write(f"Probabilidad: {probability:.2f} %")

    st.write("**************************")
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    
    if uploaded_file is not None:
        column_name = st.text_input("Ingresa el nombre de la columna a tratar:", "")
        df = pd.read_csv(uploaded_file) 
        if column_name:
            if column_name in df.columns:
                model_name = st.selectbox("Selecciona el modelo:", urlClassifier.get_models(), key= "select_box_csv") 
                if st.button("Predecir", key= "btn_csv"):
                    df_result = urlClassifier.predict_csv(model_name, df, column_name)
                    bar_chart(df_result)
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar CSV con Predicciones",
                        data=csv,
                        file_name='predicciones.csv',
                        mime='text/csv',
                    )
            else:
                st.error(f"La columna '{column_name}' no se encuentra en los datos cargados.")