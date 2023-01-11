import streamlit as st
from streamlit_option_menu import option_menu

from streamlit_modal import Modal
import streamlit.components.v1 as components


import pandas as pd

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import warnings

from models import datosBD
from CRUD import crud
from connection import connect

# Definimos las funciones usadas en la APP
def popup():
    import hydralit_components as hc
    import streamlit as st


    hc.hydralit_experimental(True)


    modal_code = """
                <div>
                <!-- Modal -->
                <div class="modal fade" id="exampleModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
                <div class="modal-dialog" role="document">
                <div class="modal-content">
                <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
                </div>
                <div class="modal-body">
                <div class="container">
                <h2>¿Desea guardar en Base de Datos?</h2>

                </div>
                </div>
                <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <button type="button" class="btn btn-primary">Guardar en BBDD</button>
                </div>
                </div>
                </div>
                </div>
                </div>
                """


    st.markdown(modal_code,unsafe_allow_html=True)
    query_param = st.experimental_get_query_params()

    if query_param:
        st.write('We caputred these values from the experimental modal form using Javascript + HTML + Streamlit + Hydralit Components.')
        st.write(query_param)

def pop_up_guardar(modal, submit, titulo,nombre_boton,respuesta_popup):
    # Opcion de formulario
    if submit:   
        modal.open()
    if modal.is_open():
        with modal.container():
            guardar_BD = st.button(nombre_boton, disabled=False) 
        if guardar_BD:
            modal.close()
            return respuesta_popup=="SI"
        else:
            modal.close()
            return respuesta_popup=="NO"

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

def menu_graphics():
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Graphics Database</p>', unsafe_allow_html=True)

    st.markdown('Podremos mostrar las principales gráficas de la base de datos')
    # Descargamos los datos
    data=pd.read_csv("media\heart.csv")

    c30, c31 = st.columns([1.5, 8.5]) 
    with c30:
        ModelType = st.radio(
                                "Elije el tipo de gráfico",
                                ["Listado","Gráfico Tarta", "Gráfico dos variables", "BoxPlot", "Histogramas"],
                                help="En esta versión sólo puedes elejir solo estos tipos de gráficos"
                            )
        if ModelType == "Gráfico Tarta":
            st.write("-"*12)
            columna="Sex"
            columnas = ("Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease")
            columna = st.selectbox("Selecciona un dato:", columnas)
        elif ModelType == "Gráfico dos variables":
            st.write("-"*12)
            columna1="Age"
            columnas1 = ("Age","RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
            columna1 = st.selectbox("Selecciona la variable continua:", columnas1, help="Las variables continuas son aquellas que puede tomar todos los valores posibles dentro de un cierto intervalo de la recta real ")

            columna2="Sex"
            columnas2 = ("Sex","ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease")
            columna2 = st.selectbox("Selecciona la variable cualitativa:", columnas2,  help="Una variable cualitativa nominal presenta modalidades no numéricas que no admiten un criterio de orden")

        elif ModelType == "BoxPlot":
            st.write("-"*12)
            columna="Age"
            columnas = ("Age", "RestingBP", "Cholesterol",  "MaxHR", "Oldpeak")
            columna = st.selectbox("Selecciona un dato:", columnas)          

        elif ModelType == "Histogramas":
            st.write("-"*12)
            checkbox_val = st.checkbox("Por 'HeartDisease'", value=True)
            columna="Age"
            columnas = ("Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope")
            columna = st.selectbox("Selecciona un dato:", columnas)    

        elif ModelType == "Listado":
            st.write("-"*12)

            number_slider = st.slider(
                                "Numero de filas a mostrar",
                                min_value=1,
                                max_value=len(data),
                                value=(0,100),
                                help="You can choose the number of rows to display. Between 0 and data length, default number is 100.",
                            )
            slider_min=number_slider[0]
            slider_max=number_slider[1]

    with c31:
        if ModelType == "Listado":
            data_query=data[slider_min:slider_max]
            gb = GridOptionsBuilder.from_dataframe(data_query)
            gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
#            gb.configure_side_bar() #Add a sidebar
            gridOptions = gb.build()
            AgGrid(
                                    data_query,
                                    gridOptions=gridOptions,
                                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                                    # theme='blue', #Add theme color to the table
                                    enable_enterprise_modules=True,
                                    height=500, 
                                    width='100%',
                                    )


        elif ModelType == "Gráfico dos variables":
            titulo = "Gráfico de las variables: " + columna1 + " y " + columna2

            fig = px.histogram(data, x=columna1, color=columna2, marginal="rug",
                        hover_data=data.columns, title=titulo)

            fig.update_layout(
                                autosize=False,
                                width=600,
                                height=600,
                                margin=dict(
                                    l=50,
                                    r=50,
                                    b=100,
                                    t=100,
                                    pad=4
                                        ),
                                paper_bgcolor="gainsboro",
                                titlefont=dict(size=30),
                                legend=dict(font = dict(size = 20)),
                                yaxis=dict(title_text="Número de Personas"),
                                font=dict(size=20)
                            )

            st.plotly_chart(fig, use_container_width=True)

        elif ModelType == "BoxPlot":
            titulo = "BoxPlot: " + columna

            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data[columna][data.HeartDisease==0],
                name="Sin enfermedad Cardiaca",
                jitter=0.3,
                pointpos=-1.8,
                boxpoints='all', # represent all points
                marker_color='rgb(7,40,89)',
                line_color='rgb(7,40,89)'
                                 ))

            fig.add_trace(go.Box(
                y=data[columna][data.HeartDisease==1],
                name="Con Enfernedad Cardiaca",
                boxpoints='all', # all points
                marker_color='rgb(107,174,214)',

                line_color='rgb(107,174,214)'
                                 ))

            fig.update_layout(title_text=titulo)
            fig.update_layout(
                            autosize=False,
                            width=600,
                            height=600,
                            margin=dict(
                                l=50,
                                r=50,
                                b=100,
                                t=100,
                                pad=4
                                    ),
                            paper_bgcolor="gainsboro",
                            titlefont=dict(size=30),
                            legend=dict(font = dict(size = 20)),
                            yaxis=dict(title_text=columna),
                            font=dict(size=20)
                            )
            st.plotly_chart(fig, use_container_width=True)



        elif ModelType == "Histogramas":
            titulo = "Histograma: " + columna
            if checkbox_val:
                fig = px.histogram(data, x=columna, 
                    color="HeartDisease", 
                    marginal="box", 
                    histnorm='probability density',  
                    color_discrete_map={0:'#0D3383',1:'#D81E1F'}, title=titulo)
            else:
                fig = px.histogram(data, x=columna, 
                    marginal="box", 
                    histnorm='probability density',  
                    color_discrete_map={0:'#0D3383',1:'#D81E1F'}, title=titulo)
            fig.update_layout(
                            autosize=False,
                            width=600,
                            height=600,
                            margin=dict(
                                l=50,
                                r=50,
                                b=100,
                                t=100,
                                pad=4
                                    ),
                            paper_bgcolor="gainsboro",
                            titlefont=dict(size=30),
                            legend=dict(font = dict(size = 20)),
                            yaxis=dict(title_text="Densidad"),
                            font=dict(size=20)
                            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            titulo = "Gráfico columna: " + columna
            colors = ['mediumturquoise', 'darkorange', 'lawngreen','lightcoral']
            fig = px.pie(data, names=columna, title=titulo)
            fig.update_traces(hoverinfo='label+percent', 
                                textinfo='percent', textfont_size=20,
                                marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            fig.update_layout(
                                autosize=False,
                                width=600,
                                height=600,
                                margin=dict(
                                    l=50,
                                    r=50,
                                    b=100,
                                    t=100,
                                    pad=4
                                        ),
                                paper_bgcolor="gainsboro",
                                titlefont=dict(size=30),
                                legend=dict(x=0, y=0.9,font = dict(size = 20))
                            )
            st.plotly_chart(fig, use_container_width=True)

def menu_query():
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
            checkbox_val = st.checkbox("Form checkbox", value=True)

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

def preparar_datos_BBDD(datosBD, Accion):
    # Preparo los datos como necesita el X_test y la BBDD:
    if datosBD.Sex == "M":
        sex_M= 1
    else:
        sex_M= 0

    if datosBD.RestingECG == "LVH":
        restingEGC_N=0
    elif datosBD.RestingECG == "Normal":    
        restingEGC_N=1
    elif datosBD.RestingECG == "ST":    
        restingEGC_N=2

    if datosBD.ChestPainType == "ASY":
        chestPainType_N=0
    elif datosBD.ChestPainType == "NAP":    
        chestPainType_N=1
    elif datosBD.ChestPainType == "ATA":    
        chestPainType_N=2
    elif datosBD.ChestPainType == "TA":    
        chestPainType_N=3

    if datosBD.ST_Slope == "Down":
        st_Slope_N=0
    elif datosBD.ST_Slope == "Flat":    
        st_Slope_N=1
    elif datosBD.ST_Slope == "Up":    
        st_Slope_N=2

    if datosBD.ExerciseAngina == "Y":
        ExerciseAngina_Y= 1
    else:
        ExerciseAngina_Y= 0

    if Accion=="ESTIMACION":
        # creamos el dato X_test para hacer la predicción
        datos = [
            {"Age":datosBD.Age, "ChestPainType_N":chestPainType_N, "RestingBP":datosBD.RestingBP, 
            "Cholesterol":datosBD.Cholesterol, "FastingBS":datosBD.FastingBS, "RestingEGC_N":restingEGC_N, 
            "MaxHR":datosBD.MaxHR, "Oldpeak":datosBD.Oldpeak, "ST_Slope_N":st_Slope_N, "Sex_M":sex_M, 
            "ExerciseAngina_Y":ExerciseAngina_Y}
            ]
        X_test=pd.DataFrame(datos)
        return X_test
    else:
        datos = [
            {"Age":datosBD.Age, "Sex":datosBD.Sex, "ChestPainType":datosBD.ChestPainType, "RestingBP":datosBD.RestingBP, 
            "Cholesterol":datosBD.Cholesterol, "FastingBS":datosBD.FastingBS, "RestingECG":datosBD.RestingECG, 
            "MaxHR":datosBD.MaxHR, "ExerciseAngina":datosBD.ExerciseAngina, "Oldpeak":datosBD.Oldpeak, "ST_Slope":datosBD.ST_Slope, 
            "HeartDisease":datosBD.HeartDisease}
            ]
        registro=pd.DataFrame(datos)
        return registro

def guardar_BBDD(registro, tipo_BBDD):
    if tipo_BBDD == "CSV":
        st.write("registro",registro)
        data=pd.read_csv("media\heart.csv")
        st.write("data",data)
        data = data.append(registro,ignore_index=True)
        data.to_csv("media\heart2.csv")
        st.write("data2",data)
    # elif tipo_BBDD == "Sqlite":
    #     st.write("registro",registro)
    #     data=pd.read_csv("media\heart.csv")
    #     st.write("data",data)
    #     data = data.append(registro,ignore_index=True)
    #     data.to_csv("media\heart2.csv")
    #     st.write("data2",data)
    # elif tipo_BBDD == "Mongo":
    #     st.write("registro",registro)
    #     data=pd.read_csv("media\heart.csv")
    #     st.write("data",data)
    #     data = data.append(registro,ignore_index=True)
    #     data.to_csv("media\heart2.csv")
    #     st.write("data2",data)
    else:
        st.write("ERROR")

def menu_prediccion():

    # Seleccion de la acción principal entre realizar una prediccion o guarda datos en BBDD
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Predicción</p>', unsafe_allow_html=True)
    ModelType =  st.radio(
                        "¿Que deseas realizar?",
                        ["Realizar una estimación", "Dar de alta un dato de un paciente"],
                    )

    if ModelType == "Realizar una estimación":
        st.markdown("El algoritmo usado para la predicción es el 'Extra Trees Classifier'. Con una precisión del modelo del 90%.")
        st.markdown("A continuación introduzca sus datos y haga clic para predecir si el paciente tiene o no una enfermedad cardiaca")
    else:
        st.markdown("A continuación introduzca sus datos y haga clic para guardar los datos de un paciente en la base de datos")


    # Formulario principal de entrada de los datos del paciente
    predict = False
    formulario_1 = st.form("form1", clear_on_submit=True)
    with formulario_1:
        col1, col2, col3, col4, col5, col6 = st.columns([0.25,2.5,2.5,2.5,2.5,0.25])
        with col2:
            datosBD.Sex = st.selectbox("Gender", ("M", "F"), help = " [M: Male, F: Female]")
        with col3:
            datosBD.Age = st.slider("Patient Age", min_value=25, max_value=90,value=50,step = 1)
        with col4:
            datosBD.ChestPainType = st.selectbox("Chest Pain Type", ("TA", "ATA", "NAP"), help = " [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]")
        with col5:
            datosBD.RestingBP = st.slider("Resting BP", min_value=60, max_value=220, value=18, step = 1)
   
        st.write(" ")

        ccol1, ccol2, ccol3, ccol4, ccol5, ccol6 = st.columns([0.25,2.5,2.5,2.5,2.5,0.25])
        with ccol2:
            datosBD.Cholesterol = st.slider("Cholesterol", min_value=60, max_value=700, value=200, step = 1)
        with ccol3:
            datosBD.FastingBS = st.selectbox("Fasting BS", (0, 1), help = "[1: if FastingBS > 120 mg/dl, 0: otherwise]")  
        with ccol4:
            datosBD.RestingECG  = st.selectbox("Resting EGC", ("Normal", "ST", "LVH"), help = "[Normal: Normal, ST: Tener anormalidad de onda ST-T (inversiones de onda T y / o elevación o Depresión de ST de > 005 mV, LVH:  Muestra hipertropía ventricular izquierda probable o definitiva según los criterios de Estes.(showing probable or definite left ventricular hypertrophy by Estes' criteria]")
        with ccol5:
            datosBD.MaxHR = st.slider("MaxHR", min_value=40, max_value=250, value=130, step = 1)

        st.write("")

        cccol1, cccol2, cccol3, cccol4, cccol5, cccol6 = st.columns([0.25,2.5,2.5,2.5,2.5,0.25])
        with cccol2:
            datosBD.Oldpeak = st.slider("Oldpeak", min_value=-3.5, max_value=7.5, value=1.0, step = 0.5)
            if ModelType == "Realizar una estimación":
                # Initialize disabled for form_submit_button to True
                guardaBD = st.radio("¿Guardar en BBDD?",
                        ["SI","NO",])
                
                predict = st.form_submit_button("Predict")

            else:
                guardar_form1 = formulario_1.form_submit_button("Guardar")
                if guardar_form1:
                    registro = preparar_datos_BBDD(datosBD,"ALTA")
                    guardar_BBDD(registro)
        with cccol3:
            datosBD.ST_Slope  = st.selectbox("ST_Slope", ("Up", "Flat", "Down"), help = "The slope of the peak exercise ST segment--> [Up: upsloping - ascenso, Flat: plano, Down: downsloping - descenso]")  
        with cccol4:
            datosBD.ExerciseAngina  = st.selectbox("Exercise Angina", ("Y", "N"), help = "[Y: Yes, N: No]")   
        with cccol5:
            if ModelType == "Dar de alta un dato de un paciente":
                datosBD.HeartDisease  = st.selectbox("Heart Disease", (0, 1), disabled=False, help = "[1: heart disease, 0: Normal]") 

        if predict:            
            X_test = preparar_datos_BBDD(datosBD,"ESTIMACION")
            # Leemos el modelo .
            with open('model_ET_pkl' , 'rb') as f:
                clf_ET = pickle.load(f)

            y_pred=clf_ET.predict(X_test)
            if y_pred[0] ==0:
                datosBD.HeartDisease = 0
                respuesta_estimacion = "NO" 
                not_y_pred=1
            else:
                datosBD.HeartDisease = 1
                respuesta_estimacion = "SI" 
                not_y_pred=0
            with cccol5:
                datosBD.HeartDisease  = st.selectbox("Heart Disease", (y_pred[0], not_y_pred), disabled=True, help = "[1: heart disease, 0: Normal]") 

            if guardaBD=="SI":
                registro = preparar_datos_BBDD(datosBD,"ALTA")
                guardar_BBDD(registro)

def main():

    warnings.filterwarnings("ignore")

    # set settings for streamlit page
    st.set_page_config(page_icon="logo.png", 
                        page_title="PMHD", 
                        layout="wide"
                        )
    # from streamlit_extras.app_logo import add_logo
    # add_logo("https://github.com/MBAristizabal/Heart-Failure/tree/develop/logo.png")

    # hide streamlit menu bar
    estilos = """
            <style>
            #MainMenu{
                visibility:visible;
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
    #  Aplicamos la configuracion a streamlit
    st.markdown(estilos, unsafe_allow_html=True)

    # Generamos Menu con option_menu
    with st.sidebar:
        choose = option_menu("App PMHD", ["About", "DataBase", "Graphics Database", "Query Database", "Predicción"],
                            icons=["house", 'device-ssd', 'graph-down', "diagram-2" , 'speedometer'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4BA3FF"},
        }
        )


    if choose == "About":
        menu_about()

    elif choose == "DataBase":
        menu_database()

    elif choose == "Graphics Database":
        menu_graphics()

    elif choose == "Query Database":
        menu_query()

    elif choose == "Predicción":
        menu_prediccion()


if __name__ == "__main__":
    main()