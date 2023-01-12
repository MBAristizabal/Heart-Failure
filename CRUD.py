import pandas as pd

from connection import connect
from models import datosBD
import settings

import sqlite3 as sql


class crud:

    def insert_registro(registro):

        nombredb=settings.DATABASE
        coleccion=settings.COLECTION

        db = connect(nombredb)
        
        registro_dic = registro.to_dict(orient="records")
        
        db[coleccion].insert_many(registro_dic)

        return registro_dic

    def create_tabla():

        # IMPORTAMOS EL CSV
        data=pd.read_csv("media\heart.csv")
    
        columnas = [
                    "Age", "Sex","ChestPainType", "RestingBP", 
                    "Cholesterol", "FastingBS", "RestingECG", 
                    "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope","HeartDisease"
                    ]

        # Generamos el DataFrame con los campos que queremos tener   
        df = pd.DataFrame(data, columns = columnas)

        num_registros = len(df)


        if num_registros != 0:

            # Lo convertimos en lista de diccionarios para cargarlo a la BBDD Mongo
            df_dic = df.to_dict('records')

            nombredb=settings.DATABASE
            coleccion=settings.COLECTION

            db = connect(nombredb)

            db[coleccion].insert_many(df_dic)
        
        return num_registros


    def read_tabla():
        # conexión a base de datos
        nombredb=settings.DATABASE
        coleccion=settings.COLECTION

        try:
            db = connect(nombredb)

        except Exception as e:
            error = "Error getting data: %s" % str(e)
            return {f"msg":error}

        datos = db[coleccion].find({})
        columnas = [
                    "Age", "Sex","ChestPainType", "RestingBP", 
                    "Cholesterol", "FastingBS", "RestingECG", 
                    "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope","HeartDisease"
                    ]
        lista = []
        for dato in datos:
            lista.append(dato)
        df = pd.DataFrame(lista, columns=columnas)
        return df


    def insertRow(registro):
        conn = sql.connect('HF.db')
        cursor = conn.cursor()
        instruccion= f"INSERT INTO HF VALUES ({int(registro.Age)},'{registro.Sex}','{registro.ChestPainType}',{int(registro.RestingBP)},{int(registro.Cholesterol)},{int(registro.FastingBS)},'{registro.RestingECG}',{int(registro.MaxHR)},'{registro.ExerciseAngina}',{float(registro.Oldpeak)},'{registro.ST_Slope}',{int(registro.HeartDisease)})"
        cursor.execute(instruccion)
        conn.commit()
        conn.close()

    def readRows():
        conn = sql.connect('HF.db')
        cursor = conn.cursor() 
        instruccion = f"SELECT * FROM HF"
        cursor.execute(instruccion)
        datos= cursor.fetchall()
        conn.commit()
        conn.close()


