import streamlit as st


from streamlit_option_menu import option_menu
# import streamlit.components.v1 as html
# from  PIL import Image
# import numpy as np
# import cv2
import pandas as pd
# from st_aggrid import AgGrid
# import plotly.express as px
# import io 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

# 1) Poner la pantalla en modo apaisado (que ocupe todo el espacio)
st.set_page_config(page_icon="logo.png", page_title="PMHD", layout="wide")

hide_menu = """
<style>
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:visible;
}
footer:after{
    content:'Copyright @ 2022: Maria Belen & Carlos Javier';
    display:block;
    position:relative;
    color:orange;
}
</style>
"""

def menu_about():
    col1, col2 = st.columns( [0.1, 0.9])
    with col1:               # To display the header text using css style
        st.image("logo.png", width=75 )
    with col2:               # To display brand log
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">PMHD - Predicion Model for Heart Disease</p>', unsafe_allow_html=True)


    st.write("PMHD, es un interfaz construido en Streamlit para analizar si un paciente puede tener una enfermedad cardiaca")    
    st.image("imagen_1.jfif", width=700 )


    # Hacemos el pie de pagina
    with st.expander("ℹ️ - About the creators", expanded=True):
        st.write("- María Belén Aristizábal and Carlos Javier Cuenca are data science professionals, enthusiasts, and bloggers. They write data science articles and tutorials on Python, data visualization, Streamlit, etc. They are also amateur dancers who love salsa music.")
        st.write("- To read data science posts, please visit their Medium blog at: https://XXXXXXXXXXXXXX")
        st.markdown("")

def menu_database():
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">DataBase</p>', unsafe_allow_html=True)

    st.subheader('Introducción de la base de datos')
    st.markdown('Descripción del origen de la base de datos y de cada uno de los campos, significado y contenido.')
    st.markdown('- **Age** : La edad del paciente.')
    st.markdown('- **Sex** : El sexo del paciente -> [M: Male, F: Female]')
    st.markdown('- **ChestPainType** : Tipo de dolor de pecho [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]')
    st.markdown("""- **RestingBP** : presión arterial en reposo:
                \n      - Tension Baja : - 90/60mmHg tenemos la tensión baja [mm Hg]
                \n      - Hipertenso: por encima de los 140/90 mmHg somos hipertensos
                \n      - Tension Normal: 120 mmHg para la sistólica y 80 mmHg
                \n      - Tension normal alta: Entre 120/80 mmHg y los 139/89mmHg
                """)
    st.markdown('- **Cholesterol** : Colesterol serico. Es el colesterol total e incluye ambos tipos: El colesterol de lipoproteína de baja densidad (LDL, por su sigla en inglés) y el colesterol de lipoproteína de alta densidad (HDL, por su sigla en inglés) o Colesterol malo (LDL) Se mide en [mm/dl]. \n Para personas de 20 años o mayores un nivel saludable es 125 a 200 mg/dL')
    st.markdown("""- **FastingBS** : FastingBS Nivel de azucar en ayunas (fasting blood sugar) [1: if FastingBS > 120 mg/dl, 0: otherwise]. Los valores de azúcar en la sangre en ayunas de:
                \n      - 99 mg/dl o menores son normales, 
                \n      - entre 100 a 125 mg/dl indican que tiene prediabetes 
                \n      - entre  126 mg/dl o mayores indican que tiene diabetes.
                """)
    st.markdown("""- **RestingECG** : Resultados electrocardiográficos en reposo (resting electrocardiogram results):
                \n      - Normal: Normal
                \n      - ST: Tener anormalidad de onda ST-T (inversiones de onda T y / o elevación o Depresión de ST de > 005 mV
                \n      - LVH:  Muestra hipertropía ventricular izquierda probable o definitiva según los criterios de Estes.(showing probable or definite left ventricular hypertrophy by Estes' criteria
                """)
    st.markdown('- **MaxHR** :frecuencia cardíaca máxima alcanzada (maximum heart rate achieved) [Numeric value between 60 and 202]')
    st.markdown('- **ExerciseAngina** : Angina inducida por ejercicio (exercise-induced angina) [Y: Yes, N: No]')
    st.markdown('- **Oldpeak** : Depresión del ST inducida por el ejercicio en relación con el descanso (oldpeak) = ST [Numeric value measured in depression]')
    st.markdown("""- **ST_Slope** : La pendiente del segmento ST de ejercicio pico (categórica con 3 niveles) - (the slope of the peak exercise ST segment)

                \n      - Up: upsloping - ascenso
                \n      - Flat: plano
                \n      - Down:  : downsloping - descenso
                """)
    st.markdown('- **HeartDisease** : Enfermedad del corazón (output class) [1: heart disease, 0: Normal]')


def main():
#    st.title("App PMHD")
    st.markdown(hide_menu, unsafe_allow_html=True)

    #"background-color": "#02ab21" - #4a5bf5  - 0019f5 - 
#4a5bf5  - 0019f5 - 
#
    with st.sidebar:
        choose = option_menu("App PMHD", ["About", "DataBase", "Graphics Database", "Query Database", "Predicción"],
                            icons=["house", 'device-ssd', 'graph-down', "diagram-2" , 'speedometer'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4a5bf5"},
        }
        )


    if choose == "About":
        menu_about()


    elif choose == "DataBase":
        menu_database()

    elif choose == "Graphics Database":

        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Graphics Database</p>', unsafe_allow_html=True)

        st.markdown('Podremos mostrar las principales gráficas de la base de datos')
        # Descargamos los datos
        data=pd.read_csv("media\heart.csv")

        st.dataframe(data.head())

        c30, c31 = st.columns([0.2, 0.8]) # 2 columnas: 10%, 30%
        with c30:
        #    st.title("Columna 31")
            columna="Age"
            columnas = ("Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope")
            columna = st.selectbox("Selecciona un campo:", columnas)
            st.write("Has elegido: ", columna)

            titulo = "Gráfico" + columna
        with c31:
            # 4) Visualizar la información usando plotly: 
            import plotly.express as px

            # data_close=data.columna

            fig = px.line(data, x="HeartDisease", y=columna, title=titulo)

            st.plotly_chart(fig, use_container_width=True)




            # with st.form("my_form"):
            #     st.write("Inside the form")
            #     slider_val = st.slider("Form slider")
            #     checkbox_val = st.checkbox("Form checkbox")

            #     ModelType = st.radio(
            #                         "Choose your model",
            #                         ["DistilBERT (Default)", "Flair"],
            #                         help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
            #                     )

            #     top_N = st.slider(
            #                         "# of results",
            #                         min_value=1,
            #                         max_value=30,
            #                         value=10,
            #                         help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
            #                     )

            #     min_Ngrams = st.number_input(
            #                                     "Minimum Ngram",
            #                                     min_value=1,
            #                                     max_value=4,
            #                                     help="""The minimum value for the ngram range.
            #                         *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
            #                         To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            #                                     # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
            #                                 )

            #     # Every form must have a submit button.
            #     submitted = st.form_submit_button("Submit")
            #     if submitted:
            #         st.write("slider", slider_val, "checkbox", checkbox_val)

            # st.write("Outside the form")


    elif choose == "Query Database":
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Query Database</p>', unsafe_allow_html=True)
        st.markdown('To start a data science project in Python, you will need to first import your data into a Pandas data frame. Often times we have our raw data stored in a local folder in csv format. Therefore let\'s learn how to use Pandas\' read_csv method to read our sample data into Python.')


        c30, c31 = st.columns([0.1, 0.3]) # 2 columnas: 10%, 30%
        with c30:
            st.title("Columna 31")
            with st.form("my_form"):
                st.write("Inside the form")
                slider_val = st.slider("Form slider")
                checkbox_val = st.checkbox("Form checkbox")

                ModelType = st.radio(
                                    "Choose your model",
                                    ["DistilBERT (Default)", "Flair"],
                                    help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
                                )

                top_N = st.slider(
                                    "# of results",
                                    min_value=1,
                                    max_value=30,
                                    value=10,
                                    help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
                                )

                min_Ngrams = st.number_input(
                                                "Minimum Ngram",
                                                min_value=1,
                                                max_value=4,
                                                help="""The minimum value for the ngram range.
                                    *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
                                    To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
                                                # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
                                            )

                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.write("slider", slider_val, "checkbox", checkbox_val)

            st.write("Outside the form")

    elif choose == "Predicción":
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Predicción</p>', unsafe_allow_html=True)
        st.markdown('To start a data science project in Python, you will need to first import your data into a Pandas data frame. Often times we have our raw data stored in a local folder in csv format. Therefore let\'s learn how to use Pandas\' read_csv method to read our sample data into Python.')

        # Leemos el modelo .
    #    with open('model_ET_pkl' , 'rb') as f:
    #        clf_ET = pickle.load(f)


















if __name__ == "__main__":
    main()