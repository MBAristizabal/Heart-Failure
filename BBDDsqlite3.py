import sqlite3 as sql

def createBD():
    conn = sql.connect('HF.db')
    conn.commit()
    conn.close()


def createTable():
    conn = sql.connect('HF.db')
    cursor = conn.cursor()
    cursor.execute(
        """ CREATE TABLE HF (
                Age int,
                Sex str,
                ChestPainType str,
                RestingBP int,
                Cholesterol int,
                FastingBS int,
                RestingEGC str,
                MaxHR int,
                ExerciseAngina str,
                Oldpeak float,
                ST_Slope str,
                HeartDisease int)""")
    conn.commit()
    conn.close()

def insertRow(registro):
    conn = sql.connect('HF.db')
    cursor = conn.cursor()
    instruccion= f"INSERT INTO HF VALUES ({registro.Age},'{registro.Sex}','{registro.ChestPainType}',{registro.RestingBP},{registro.Cholesterol},{registro.FastingBS},'{registro.RestingEGC}',{registro.MaxHR},'{registro.ExerciseAngina}',{registro.Oldpeak},'{registro.ST_Slope}',{registro.HeartDisease})"
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
    print(datos)

def insertRows(HFList):
    conn = sql.connect('HF.db')
    cursor = conn.cursor() 
    instruccion = f"INSERT INTO HF VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
    cursor.executemany(instruccion, HFList)
    conn.commit()
    conn.close()


def search():
    conn = sql.connect('HF.db')
    cursor = conn.cursor() 
    instruccion = f"SELECT * FROM HF WHERE Cholesterol >200"
    cursor.execute(instruccion)
    datos= cursor.fetchall()
    conn.commit()
    conn.close()
    print(datos)

def updateFields():
    conn = sql.connect('HF.db')
    cursor = conn.cursor() 
    instruccion = f"UPDATE HF SET  HeartDisease = 1 WHERE Cholesterol >220"
    cursor.execute(instruccion)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    #createBD()
    #createTable()
    #insertRow('30', 'F', 'ATA','144', '240','0','ST','150','N','1.0','UP', '0')
    #readRows()

    HF = [
        (60, 'F', 'NAP',124, 230,0,'ST',150,'N',1.7,'UP', 0),
        (77, 'M', 'ATA',175, 209,1,'ST',150,'Y',1.6,'UP', 1),
        (75, 'M', 'ASY',144, 243,0,'ST',150,'N',1.4,'UP', 0),
        (80, 'M', 'ATA',133, 220,1,'ST',150,'Y',0.9,'UP', 1),
        (90, 'F', 'NAP',158, 218,0,'ST',150,'N',0.8,'UP', 0)
    ]
    #insertRows(HF)
    #search()
    #updateFields()
    
